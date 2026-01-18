from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def generate_rp_config(self):
    rp_spec = vim.ResourceConfigSpec()
    cpu_alloc = vim.ResourceAllocationInfo()
    cpu_alloc.expandableReservation = self.cpu_expandable_reservations
    cpu_alloc.limit = self.cpu_limit
    cpu_alloc.reservation = self.cpu_reservation
    cpu_alloc_shares = vim.SharesInfo()
    if self.cpu_shares == 'custom':
        cpu_alloc_shares.shares = self.cpu_allocation_shares
    cpu_alloc_shares.level = self.cpu_shares
    cpu_alloc.shares = cpu_alloc_shares
    rp_spec.cpuAllocation = cpu_alloc
    mem_alloc = vim.ResourceAllocationInfo()
    mem_alloc.limit = self.mem_limit
    mem_alloc.expandableReservation = self.mem_expandable_reservations
    mem_alloc.reservation = self.mem_reservation
    mem_alloc_shares = vim.SharesInfo()
    if self.mem_shares == 'custom':
        mem_alloc_shares.shares = self.mem_allocation_shares
    mem_alloc_shares.level = self.mem_shares
    mem_alloc.shares = mem_alloc_shares
    rp_spec.memoryAllocation = mem_alloc
    return rp_spec