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
def build_restart_policy(self):
    restart_policy_args = {}
    if self.restart_policy is not None:
        restart_policy_args['condition'] = self.restart_policy
    if self.restart_policy_delay is not None:
        restart_policy_args['delay'] = self.restart_policy_delay
    if self.restart_policy_attempts is not None:
        restart_policy_args['max_attempts'] = self.restart_policy_attempts
    if self.restart_policy_window is not None:
        restart_policy_args['window'] = self.restart_policy_window
    return types.RestartPolicy(**restart_policy_args) if restart_policy_args else None