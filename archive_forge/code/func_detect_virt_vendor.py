from __future__ import (absolute_import, division, print_function)
import re
def detect_virt_vendor(self, key):
    virtual_vendor_facts = {}
    host_tech = set()
    guest_tech = set()
    self.detect_sysctl()
    if self.sysctl_path:
        rc, out, err = self.module.run_command('%s -n %s' % (self.sysctl_path, key))
        if rc == 0:
            if out.rstrip() == 'QEMU':
                guest_tech.add('kvm')
                virtual_vendor_facts['virtualization_type'] = 'kvm'
                virtual_vendor_facts['virtualization_role'] = 'guest'
            if out.rstrip() == 'OpenBSD':
                guest_tech.add('vmm')
                virtual_vendor_facts['virtualization_type'] = 'vmm'
                virtual_vendor_facts['virtualization_role'] = 'guest'
    virtual_vendor_facts['virtualization_tech_guest'] = guest_tech
    virtual_vendor_facts['virtualization_tech_host'] = host_tech
    return virtual_vendor_facts