from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def normalize_mac(proposed_mac, module):
    if proposed_mac is None:
        return ''
    try:
        if '-' in proposed_mac:
            splitted_mac = proposed_mac.split('-')
            if len(splitted_mac) != 6:
                raise ValueError
            for octect in splitted_mac:
                if len(octect) != 2:
                    raise ValueError
        elif '.' in proposed_mac:
            splitted_mac = []
            splitted_dot_mac = proposed_mac.split('.')
            if len(splitted_dot_mac) != 3:
                raise ValueError
            for octect in splitted_dot_mac:
                if len(octect) > 4:
                    raise ValueError
                else:
                    octect_len = len(octect)
                    padding = 4 - octect_len
                    splitted_mac.append(octect.zfill(padding + 1))
        elif ':' in proposed_mac:
            splitted_mac = proposed_mac.split(':')
            if len(splitted_mac) != 6:
                raise ValueError
            for octect in splitted_mac:
                if len(octect) != 2:
                    raise ValueError
        else:
            raise ValueError
    except ValueError:
        module.fail_json(msg='Invalid MAC address format', proposed_mac=proposed_mac)
    joined_mac = ''.join(splitted_mac)
    mac = [joined_mac[i:i + 4] for i in range(0, len(joined_mac), 4)]
    return '.'.join(mac).upper()