from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def build_ipmctl_creation_opts(self, skt=None):
    ipmctl_opts = []
    if skt:
        appdirect = skt['appdirect']
        memmode = skt['memorymode']
        reserved = skt['reserved']
        socket_id = skt['id']
        ipmctl_opts += ['-socket', socket_id]
    else:
        appdirect = self.appdirect
        memmode = self.memmode
        reserved = self.reserved
    if reserved is None:
        res = 100 - memmode - appdirect
        ipmctl_opts += ['memorymode=%d' % memmode, 'reserved=%d' % res]
    else:
        ipmctl_opts += ['memorymode=%d' % memmode, 'reserved=%d' % reserved]
    if self.interleaved:
        ipmctl_opts += ['PersistentMemoryType=AppDirect']
    else:
        ipmctl_opts += ['PersistentMemoryType=AppDirectNotInterleaved']
    return ipmctl_opts