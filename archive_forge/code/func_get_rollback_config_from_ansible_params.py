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
def get_rollback_config_from_ansible_params(params):
    if params['rollback_config'] is None:
        return None
    rollback_config = params['rollback_config'] or {}
    delay = get_nanoseconds_from_raw_option('rollback_config.delay', rollback_config.get('delay'))
    monitor = get_nanoseconds_from_raw_option('rollback_config.monitor', rollback_config.get('monitor'))
    return {'parallelism': rollback_config.get('parallelism'), 'delay': delay, 'failure_action': rollback_config.get('failure_action'), 'monitor': monitor, 'max_failure_ratio': rollback_config.get('max_failure_ratio'), 'order': rollback_config.get('order')}