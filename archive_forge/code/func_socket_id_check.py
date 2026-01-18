from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def socket_id_check(self):
    command = ['show', '-o', 'nvmxml', '-socket']
    out = self.pmem_run_ipmctl(command)
    sockets_dict = xmltodict.parse(out, dict_constructor=dict)['SocketList']['Socket']
    socket_ids = []
    for sl in sockets_dict:
        socket_ids.append(int(sl['SocketID'], 16))
    for skt in self.socket:
        if skt['id'] not in socket_ids:
            return 'Invalid socket number: %d' % skt['id']
    return None