from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.ios import (
def edit_config_or_macro(connection, commands):
    if commands[0].startswith('macro name'):
        connection.edit_macro(candidate=commands)
    else:
        connection.edit_config(candidate=commands)