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
def build_rollback_config(self):
    if self.rollback_config is None:
        return None
    rollback_config_options = ['parallelism', 'delay', 'failure_action', 'monitor', 'max_failure_ratio', 'order']
    rollback_config_args = {}
    for option in rollback_config_options:
        value = self.rollback_config.get(option)
        if value is not None:
            rollback_config_args[option] = value
    return types.RollbackConfig(**rollback_config_args) if rollback_config_args else None