from __future__ import (absolute_import, division, print_function)
from abc import ABC
import types
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
def get_option_and_origin(self, option, hostvars=None):
    try:
        option_value, origin = C.config.get_config_value_and_origin(option, plugin_type=self.plugin_type, plugin_name=self._load_name, variables=hostvars)
    except AnsibleError as e:
        raise KeyError(to_native(e))
    return (option_value, origin)