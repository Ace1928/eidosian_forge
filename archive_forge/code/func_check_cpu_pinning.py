from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def check_cpu_pinning():
    if self.param('cpu_pinning'):
        current = []
        if entity.cpu.cpu_tune:
            current = [(str(pin.cpu_set), int(pin.vcpu)) for pin in entity.cpu.cpu_tune.vcpu_pins]
        passed = [(str(pin['cpu']), int(pin['vcpu'])) for pin in self.param('cpu_pinning')]
        return sorted(current) == sorted(passed)
    return True