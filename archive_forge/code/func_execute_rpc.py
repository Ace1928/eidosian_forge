from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.plugin_utils.netconf_base import (
def execute_rpc(self, name):
    """
        RPC to be execute on remote device
        :param name: Name of rpc in string format
        :return: Received rpc response from remote host
        """
    return self.rpc(name)