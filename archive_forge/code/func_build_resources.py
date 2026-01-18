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
def build_resources(self):
    resources_args = {}
    if self.limit_cpu is not None:
        resources_args['cpu_limit'] = int(self.limit_cpu * 1000000000.0)
    if self.limit_memory is not None:
        resources_args['mem_limit'] = self.limit_memory
    if self.reserve_cpu is not None:
        resources_args['cpu_reservation'] = int(self.reserve_cpu * 1000000000.0)
    if self.reserve_memory is not None:
        resources_args['mem_reservation'] = self.reserve_memory
    return types.Resources(**resources_args) if resources_args else None