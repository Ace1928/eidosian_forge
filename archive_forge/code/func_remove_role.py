from __future__ import (absolute_import, division, print_function)
import os
import ansible.constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.yaml import yaml_load
def remove_role(self, role_name):
    del self.roles[role_name]