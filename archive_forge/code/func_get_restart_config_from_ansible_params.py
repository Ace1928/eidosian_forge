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
@staticmethod
def get_restart_config_from_ansible_params(params):
    restart_config = params['restart_config'] or {}
    condition = get_value('condition', restart_config)
    delay = get_value('delay', restart_config)
    delay = get_nanoseconds_from_raw_option('restart_policy_delay', delay)
    max_attempts = get_value('max_attempts', restart_config)
    window = get_value('window', restart_config)
    window = get_nanoseconds_from_raw_option('restart_policy_window', window)
    return {'restart_policy': condition, 'restart_policy_delay': delay, 'restart_policy_attempts': max_attempts, 'restart_policy_window': window}