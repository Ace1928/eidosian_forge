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
def get_update_config_from_ansible_params(params):
    update_config = params['update_config'] or {}
    parallelism = get_value('parallelism', update_config)
    delay = get_value('delay', update_config)
    delay = get_nanoseconds_from_raw_option('update_delay', delay)
    failure_action = get_value('failure_action', update_config)
    monitor = get_value('monitor', update_config)
    monitor = get_nanoseconds_from_raw_option('update_monitor', monitor)
    max_failure_ratio = get_value('max_failure_ratio', update_config)
    order = get_value('order', update_config)
    return {'update_parallelism': parallelism, 'update_delay': delay, 'update_failure_action': failure_action, 'update_monitor': monitor, 'update_max_failure_ratio': max_failure_ratio, 'update_order': order}