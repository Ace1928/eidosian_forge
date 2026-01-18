from __future__ import absolute_import, division, print_function
import time
import socket
import re
import json
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, EntityCollection
from ansible.module_utils.connection import Connection, exec_command
from ansible.module_utils.connection import ConnectionError
def run_cnos_commands(module, commands, check_rc=True):
    retVal = ''
    enter_config = {'command': 'configure terminal', 'prompt': None, 'answer': None}
    exit_config = {'command': 'end', 'prompt': None, 'answer': None}
    commands.insert(0, enter_config)
    commands.append(exit_config)
    for cmd in commands:
        retVal = retVal + '>> ' + cmd['command'] + '\n'
    try:
        responses = run_commands(module, commands, check_rc)
        for response in responses:
            retVal = retVal + '<< ' + response + '\n'
    except Exception as e:
        errMsg = ''
        if hasattr(e, 'message'):
            errMsg = e.message
        else:
            errMsg = str(e)
        if 'VLAN_ACCESS_MAP' in errMsg:
            return retVal + '<<' + errMsg + '\n'
        if 'confederation identifier' in errMsg:
            return retVal + '<<' + errMsg + '\n'
        retVal = retVal + '<< ' + 'Error-101 ' + errMsg + '\n'
    return str(retVal)