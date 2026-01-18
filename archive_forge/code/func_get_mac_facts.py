from __future__ import (absolute_import, division, print_function)
import struct
import time
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.sysctl import get_sysctl
def get_mac_facts(self):
    mac_facts = {}
    rc, out, err = self.module.run_command('sysctl hw.model')
    if rc == 0:
        mac_facts['model'] = mac_facts['product_name'] = out.splitlines()[-1].split()[1]
    mac_facts['osversion'] = self.sysctl['kern.osversion']
    mac_facts['osrevision'] = self.sysctl['kern.osrevision']
    return mac_facts