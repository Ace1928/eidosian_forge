from __future__ import absolute_import, division, print_function
import json
import re
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.junos import (
def filter_delete_statements(module, candidate):
    reply = get_configuration(module, format='set')
    match = reply.find('.//configuration-set')
    if match is None:
        return candidate
    config = to_native(match.text, encoding='latin-1')
    modified_candidate = candidate[:]
    for index, line in reversed(list(enumerate(candidate))):
        if line.startswith('delete'):
            newline = re.sub('^delete', 'set', line)
            if newline not in config:
                del modified_candidate[index]
    return modified_candidate