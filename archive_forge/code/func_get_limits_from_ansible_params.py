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
def get_limits_from_ansible_params(params):
    limits = params['limits'] or {}
    cpus = get_value('cpus', limits)
    memory = get_value('memory', limits)
    if memory is not None:
        try:
            memory = human_to_bytes(memory)
        except ValueError as exc:
            raise Exception('Failed to convert limit_memory to bytes: %s' % exc)
    return {'limit_cpu': cpus, 'limit_memory': memory}