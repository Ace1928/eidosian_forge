from __future__ import absolute_import, division, print_function
import re
import time
import string
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.network import is_mac
from ansible.module_utils._text import to_text, to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
from ansible_collections.community.vmware.plugins.module_utils.vmware_spbm import SPBM
def configure_cpu_and_memory(self, vm_obj, vm_creation=False):
    num_cpus = self.params['hardware']['num_cpus']
    if num_cpus is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and (not self.module.check_mode):
            if not vm_obj.config.cpuHotRemoveEnabled and num_cpus < vm_obj.config.hardware.numCPU:
                self.module.fail_json(msg='Configured cpu number is less than the cpu number of the VM, cpuHotRemove is not enabled')
            if not vm_obj.config.cpuHotAddEnabled and num_cpus > vm_obj.config.hardware.numCPU:
                self.module.fail_json(msg='Configured cpu number is more than the cpu number of the VM, cpuHotAdd is not enabled')
        num_cpu_cores_per_socket = self.params['hardware']['num_cpu_cores_per_socket']
        if num_cpu_cores_per_socket is not None:
            if num_cpus % num_cpu_cores_per_socket != 0:
                self.module.fail_json(msg='hardware.num_cpus attribute should be a multiple of hardware.num_cpu_cores_per_socket')
            if vm_obj is None or num_cpu_cores_per_socket != vm_obj.config.hardware.numCoresPerSocket:
                self.change_detected = True
                self.configspec.numCoresPerSocket = num_cpu_cores_per_socket
        if vm_obj is None or num_cpus != vm_obj.config.hardware.numCPU:
            self.change_detected = True
            self.configspec.numCPUs = num_cpus
    elif vm_creation and (not self.params['template']):
        self.module.fail_json(msg='hardware.num_cpus attribute is mandatory for VM creation')
    memory_mb = self.params['hardware']['memory_mb']
    if memory_mb is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn:
            if vm_obj.config.memoryHotAddEnabled and memory_mb < vm_obj.config.hardware.memoryMB:
                self.module.fail_json(msg='Configured memory is less than memory size of the VM, operation is not supported')
            elif not vm_obj.config.memoryHotAddEnabled and memory_mb != vm_obj.config.hardware.memoryMB and (not self.module.check_mode):
                self.module.fail_json(msg='memoryHotAdd is not enabled')
        if vm_obj is None or memory_mb != vm_obj.config.hardware.memoryMB:
            self.change_detected = True
            self.configspec.memoryMB = memory_mb
    elif vm_creation and (not self.params['template']):
        self.module.fail_json(msg='hardware.memory_mb attribute is mandatory for VM creation')
    hotadd_memory = self.params['hardware']['hotadd_memory']
    if hotadd_memory is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and (vm_obj.config.memoryHotAddEnabled != hotadd_memory) and (not self.module.check_mode):
            self.module.fail_json(msg='Configure hotadd memory operation is not supported when VM is power on')
        if vm_obj is None or hotadd_memory != vm_obj.config.memoryHotAddEnabled:
            self.change_detected = True
            self.configspec.memoryHotAddEnabled = hotadd_memory
    hotadd_cpu = self.params['hardware']['hotadd_cpu']
    if hotadd_cpu is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and (vm_obj.config.cpuHotAddEnabled != hotadd_cpu) and (not self.module.check_mode):
            self.module.fail_json(msg='Configure hotadd cpu operation is not supported when VM is power on')
        if vm_obj is None or hotadd_cpu != vm_obj.config.cpuHotAddEnabled:
            self.change_detected = True
            self.configspec.cpuHotAddEnabled = hotadd_cpu
    hotremove_cpu = self.params['hardware']['hotremove_cpu']
    if hotremove_cpu is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and (vm_obj.config.cpuHotRemoveEnabled != hotremove_cpu) and (not self.module.check_mode):
            self.module.fail_json(msg='Configure hotremove cpu operation is not supported when VM is power on')
        if vm_obj is None or hotremove_cpu != vm_obj.config.cpuHotRemoveEnabled:
            self.change_detected = True
            self.configspec.cpuHotRemoveEnabled = hotremove_cpu
    memory_reservation_lock = self.params['hardware']['memory_reservation_lock']
    if memory_reservation_lock is not None:
        if vm_obj is None or memory_reservation_lock != vm_obj.config.memoryReservationLockedToMax:
            self.change_detected = True
            self.configspec.memoryReservationLockedToMax = memory_reservation_lock
    vpmc_enabled = self.params['hardware']['vpmc_enabled']
    if vpmc_enabled is not None:
        if vm_obj and vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn and (vm_obj.config.vPMCEnabled != vpmc_enabled) and (not self.module.check_mode):
            self.module.fail_json(msg='Configure vPMC cpu operation is not supported when VM is power on')
        if vm_obj is None or vpmc_enabled != vm_obj.config.vPMCEnabled:
            self.change_detected = True
            self.configspec.vPMCEnabled = vpmc_enabled
    boot_firmware = self.params['hardware']['boot_firmware']
    if boot_firmware is not None:
        if vm_obj is not None:
            return
        self.configspec.firmware = boot_firmware
        self.change_detected = True