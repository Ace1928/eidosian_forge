import logging
import threading
from enum import Enum
def next_frozen_vm(self):
    if not self.vms:
        self.vms = self.frozen_vms_resource_pool.vm
        if len(self.vms) <= 0:
            raise ValueError(f'No vm in resource pool {self.frozen_vms_resource_pool}!')
    with self.lock:
        logger.debug('current_vm_index=%d', self.current_vm_index)
        vm = self.vms[self.current_vm_index]
        self.current_vm_index += 1
        if self.current_vm_index >= len(self.vms):
            self.current_vm_index = 0
    return vm