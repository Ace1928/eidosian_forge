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
def build_log_driver(self):
    log_driver_args = {}
    if self.log_driver is not None:
        log_driver_args['name'] = self.log_driver
    if self.log_driver_options is not None:
        log_driver_args['options'] = self.log_driver_options
    return types.DriverConfig(**log_driver_args) if log_driver_args else None