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
def build_update_config(self):
    update_config_args = {}
    if self.update_parallelism is not None:
        update_config_args['parallelism'] = self.update_parallelism
    if self.update_delay is not None:
        update_config_args['delay'] = self.update_delay
    if self.update_failure_action is not None:
        update_config_args['failure_action'] = self.update_failure_action
    if self.update_monitor is not None:
        update_config_args['monitor'] = self.update_monitor
    if self.update_max_failure_ratio is not None:
        update_config_args['max_failure_ratio'] = self.update_max_failure_ratio
    if self.update_order is not None:
        update_config_args['order'] = self.update_order
    return types.UpdateConfig(**update_config_args) if update_config_args else None