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
def rsync_down(self, source, target, docker_mount_if_possible=False):
    options = {}
    options['docker_mount_if_possible'] = docker_mount_if_possible
    options['rsync_exclude'] = self.rsync_options.get('rsync_exclude')
    options['rsync_filter'] = self.rsync_options.get('rsync_filter')
    self.cmd_runner.run_rsync_down(source, target, options=options)
    cli_logger.verbose('`rsync`ed {} (remote) to {} (local)', cf.bold(source), cf.bold(target))