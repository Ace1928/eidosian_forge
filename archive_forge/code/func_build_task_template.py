from __future__ import absolute_import, division, print_function
import shlex
import time
import traceback
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible.module_utils.basic import human_to_bytes
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text, to_native
def build_task_template(self, container_spec, placement=None):
    log_driver = self.build_log_driver()
    restart_policy = self.build_restart_policy()
    resources = self.build_resources()
    task_template_args = {}
    if placement is not None:
        task_template_args['placement'] = placement
    if log_driver is not None:
        task_template_args['log_driver'] = log_driver
    if restart_policy is not None:
        task_template_args['restart_policy'] = restart_policy
    if resources is not None:
        task_template_args['resources'] = resources
    if self.force_update:
        task_template_args['force_update'] = self.force_update
    if self.can_use_task_template_networks:
        networks = self.build_networks()
        if networks:
            task_template_args['networks'] = networks
    return types.TaskTemplate(container_spec=container_spec, **task_template_args)