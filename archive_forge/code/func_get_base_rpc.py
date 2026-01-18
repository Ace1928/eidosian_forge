from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.plugins import AnsiblePlugin
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_text
def get_base_rpc(self):
    """Returns list of base rpc method supported by remote device"""
    return self.__rpc__