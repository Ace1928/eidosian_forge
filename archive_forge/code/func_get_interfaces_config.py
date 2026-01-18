from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list, ComplexList
from ansible.module_utils.connection import Connection, ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
def get_interfaces_config(module):
    config = get_config(module)
    lines = config.split('\n')
    interfaces = {}
    interface = None
    for line in lines:
        if line == 'exit':
            if interface:
                interfaces[interface[0]] = interface
                interface = None
        elif interface:
            interface.append(line)
        else:
            match = re.match('^interface (.*)$', line)
            if match:
                interface = list()
                interface.append(line)
    return interfaces