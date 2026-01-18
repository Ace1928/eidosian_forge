from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
def get_uptime_facts(self):
    sysctl_cmd = self.module.get_bin_path('sysctl')
    cmd = [sysctl_cmd, '-b', 'kern.boottime']
    rc, out, err = self.module.run_command(cmd, encoding=None)
    struct_format = '@L'
    struct_size = struct.calcsize(struct_format)
    if rc != 0 or len(out) < struct_size:
        return {}
    kern_boottime, = struct.unpack(struct_format, out[:struct_size])
    return {'uptime_seconds': int(time.time() - kern_boottime)}