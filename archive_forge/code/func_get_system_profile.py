from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
def get_system_profile(self):
    rc, out, err = self.module.run_command(['/usr/sbin/system_profiler', 'SPHardwareDataType'])
    if rc != 0:
        return dict()
    system_profile = dict()
    for line in out.splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            system_profile[key.strip()] = ' '.join(value.strip().split())
    return system_profile