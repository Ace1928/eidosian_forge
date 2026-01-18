import logging
import os
import subprocess
import time
import traceback
from threading import Thread
import click
from ray._private.usage import usage_constants, usage_lib
from ray.autoscaler._private import subprocess_output_util as cmd_output_util
from ray.autoscaler._private.cli_logger import cf, cli_logger
from ray.autoscaler._private.command_runner import (
from ray.autoscaler._private.constants import (
from ray.autoscaler._private.event_system import CreateClusterEvent, global_event_system
from ray.autoscaler._private.log_timer import LogTimer
from ray.autoscaler.tags import (
def do_update(self):
    self.provider.set_node_tags(self.node_id, {TAG_RAY_NODE_STATUS: STATUS_WAITING_FOR_SSH})
    cli_logger.labeled_value('New status', STATUS_WAITING_FOR_SSH)
    deadline = time.time() + AUTOSCALER_NODE_START_WAIT_S
    self.wait_ready(deadline)
    global_event_system.execute_callback(CreateClusterEvent.ssh_control_acquired)
    node_tags = self.provider.node_tags(self.node_id)
    logger.debug('Node tags: {}'.format(str(node_tags)))
    if self.provider_type == 'aws' and self.provider.provider_config:
        from ray.autoscaler._private.aws.cloudwatch.cloudwatch_helper import CloudwatchHelper
        CloudwatchHelper(self.provider.provider_config, self.node_id, self.provider.cluster_name).update_from_config(self.is_head_node)
    if node_tags.get(TAG_RAY_RUNTIME_CONFIG) == self.runtime_hash:
        init_required = self.cmd_runner.run_init(as_head=self.is_head_node, file_mounts=self.file_mounts, sync_run_yet=False)
        if init_required:
            node_tags[TAG_RAY_RUNTIME_CONFIG] += '-invalidate'
            self.restart_only = False
    if self.restart_only:
        self.setup_commands = []
    if node_tags.get(TAG_RAY_RUNTIME_CONFIG) == self.runtime_hash and (not self.file_mounts_contents_hash or node_tags.get(TAG_RAY_FILE_MOUNTS_CONTENTS) == self.file_mounts_contents_hash):
        cli_logger.print('Configuration already up to date, skipping file mounts, initalization and setup commands.', _numbered=('[]', '2-6', NUM_SETUP_STEPS))
    else:
        cli_logger.print('Updating cluster configuration.', _tags=dict(hash=self.runtime_hash))
        self.provider.set_node_tags(self.node_id, {TAG_RAY_NODE_STATUS: STATUS_SYNCING_FILES})
        cli_logger.labeled_value('New status', STATUS_SYNCING_FILES)
        self.sync_file_mounts(self.rsync_up, step_numbers=(1, NUM_SETUP_STEPS))
        if node_tags.get(TAG_RAY_RUNTIME_CONFIG) != self.runtime_hash:
            self.provider.set_node_tags(self.node_id, {TAG_RAY_NODE_STATUS: STATUS_SETTING_UP})
            cli_logger.labeled_value('New status', STATUS_SETTING_UP)
            if self.initialization_commands:
                with cli_logger.group('Running initialization commands', _numbered=('[]', 4, NUM_SETUP_STEPS)):
                    global_event_system.execute_callback(CreateClusterEvent.run_initialization_cmd)
                    with LogTimer(self.log_prefix + 'Initialization commands', show_status=True):
                        for cmd in self.initialization_commands:
                            global_event_system.execute_callback(CreateClusterEvent.run_initialization_cmd, {'command': cmd})
                            try:
                                self.cmd_runner.run(cmd, ssh_options_override_ssh_key=self.auth_config.get('ssh_private_key'), run_env='host')
                            except ProcessRunnerError as e:
                                if e.msg_type == 'ssh_command_failed':
                                    cli_logger.error('Failed.')
                                    cli_logger.error('See above for stderr.')
                                raise click.ClickException('Initialization command failed.') from None
            else:
                cli_logger.print('No initialization commands to run.', _numbered=('[]', 4, NUM_SETUP_STEPS))
            with cli_logger.group('Initializing command runner', _numbered=('[]', 5, NUM_SETUP_STEPS)):
                self.cmd_runner.run_init(as_head=self.is_head_node, file_mounts=self.file_mounts, sync_run_yet=True)
            if self.setup_commands:
                with cli_logger.group('Running setup commands', _numbered=('[]', 6, NUM_SETUP_STEPS)):
                    global_event_system.execute_callback(CreateClusterEvent.run_setup_cmd)
                    with LogTimer(self.log_prefix + 'Setup commands', show_status=True):
                        total = len(self.setup_commands)
                        for i, cmd in enumerate(self.setup_commands):
                            global_event_system.execute_callback(CreateClusterEvent.run_setup_cmd, {'command': cmd})
                            if cli_logger.verbosity == 0 and len(cmd) > 30:
                                cmd_to_print = cf.bold(cmd[:30]) + '...'
                            else:
                                cmd_to_print = cf.bold(cmd)
                            cli_logger.print('{}', cmd_to_print, _numbered=('()', i, total))
                            try:
                                self.cmd_runner.run(cmd, run_env='auto')
                            except ProcessRunnerError as e:
                                if e.msg_type == 'ssh_command_failed':
                                    cli_logger.error('Failed.')
                                    cli_logger.error('See above for stderr.')
                                raise click.ClickException('Setup command failed.')
            else:
                cli_logger.print('No setup commands to run.', _numbered=('[]', 6, NUM_SETUP_STEPS))
    with cli_logger.group('Starting the Ray runtime', _numbered=('[]', 7, NUM_SETUP_STEPS)):
        global_event_system.execute_callback(CreateClusterEvent.start_ray_runtime)
        with LogTimer(self.log_prefix + 'Ray start commands', show_status=True):
            for cmd in self.ray_start_commands:
                env_vars = {}
                if self.is_head_node:
                    if usage_lib.usage_stats_enabled():
                        env_vars[usage_constants.USAGE_STATS_ENABLED_ENV_VAR] = 1
                    else:
                        env_vars[usage_constants.USAGE_STATS_ENABLED_ENV_VAR] = 0
                if self.provider_type != 'local':
                    if self.node_resources:
                        env_vars[RESOURCES_ENVIRONMENT_VARIABLE] = self.node_resources
                    if self.node_labels:
                        env_vars[LABELS_ENVIRONMENT_VARIABLE] = self.node_labels
                try:
                    old_redirected = cmd_output_util.is_output_redirected()
                    cmd_output_util.set_output_redirected(False)
                    self.cmd_runner.run(cmd, environment_variables=env_vars, run_env='auto')
                    cmd_output_util.set_output_redirected(old_redirected)
                except ProcessRunnerError as e:
                    if e.msg_type == 'ssh_command_failed':
                        cli_logger.error('Failed.')
                        cli_logger.error('See above for stderr.')
                    raise click.ClickException('Start command failed.')
        global_event_system.execute_callback(CreateClusterEvent.start_ray_runtime_completed)