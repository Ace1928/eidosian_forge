from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_create_memory_allocation(self, skt=None):

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

    def is_allocation_good(self, ipmctl_out, command):
        warning = re.compile('WARNING')
        error = re.compile('.*Error.*')
        ignore_error = re.compile('Do you want to continue? [y/n] Error: Invalid data input.')
        errmsg = ''
        rc = True
        for line in ipmctl_out.splitlines():
            if warning.match(line):
                errmsg = '%s (command: %s)' % (line, command)
                rc = False
                break
            elif error.match(line):
                if not ignore_error:
                    errmsg = '%s (command: %s)' % (line, command)
                    rc = False
                    break
        return (rc, errmsg)

    def get_allocation_result(self, goal, skt=None):
        ret = {'appdirect': 0, 'memorymode': 0}
        if skt:
            ret['socket'] = skt['id']
        out = xmltodict.parse(goal, dict_constructor=dict)['ConfigGoalList']['ConfigGoal']
        for entry in out:
            if skt and skt['id'] != int(entry['SocketID'], 16):
                continue
            for key, v in entry.items():
                if key == 'MemorySize':
                    ret['memorymode'] += int(v.split()[0])
                elif key == 'AppDirect1Size' or key == 'AapDirect2Size':
                    ret['appdirect'] += int(v.split()[0])
        capacity = self.pmem_get_capacity(skt)
        ret['reserved'] = capacity - ret['appdirect'] - ret['memorymode']
        return ret
    reboot_required = False
    ipmctl_opts = build_ipmctl_creation_opts(self, skt)
    command = ['create', '-goal'] + ipmctl_opts
    out = self.pmem_run_ipmctl(command, returnCheck=False)
    rc, errmsg = is_allocation_good(self, out, command)
    if rc is False:
        return (reboot_required, {}, errmsg)
    command = ['create', '-u', 'B', '-o', 'nvmxml', '-force', '-goal'] + ipmctl_opts
    goal = self.pmem_run_ipmctl(command)
    ret = get_allocation_result(self, goal, skt)
    reboot_required = True
    return (reboot_required, ret, '')