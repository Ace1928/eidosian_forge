from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import get_all_objs, vmware_argument_spec, find_datacenter_by_name, \
from ansible.module_utils.basic import AnsibleModule
def state_update_existing_pr(self):
    changed = False
    rp_spec = self.generate_rp_config()
    if self.mem_shares and self.mem_shares != self.resource_pool_obj.config.memoryAllocation.shares.level:
        changed = True
        rp_spec.memoryAllocation.shares.level = self.mem_shares
    if self.mem_allocation_shares and self.mem_shares == 'custom':
        if self.mem_allocation_shares != self.resource_pool_obj.config.memoryAllocation.shares.shares:
            changed = True
            rp_spec.memoryAllocation.shares.shares = self.mem_allocation_shares
    if self.mem_limit and self.mem_limit != self.resource_pool_obj.config.memoryAllocation.limit:
        changed = True
        rp_spec.memoryAllocation.limit = self.mem_limit
    if self.mem_reservation and self.mem_reservation != self.resource_pool_obj.config.memoryAllocation.reservation:
        changed = True
        rp_spec.memoryAllocation.reservation = self.mem_reservation
    if self.mem_expandable_reservations != self.resource_pool_obj.config.memoryAllocation.expandableReservation:
        changed = True
        rp_spec.memoryAllocation.expandableReservation = self.mem_expandable_reservations
    if self.cpu_shares and self.cpu_shares != self.resource_pool_obj.config.cpuAllocation.shares.level:
        changed = True
        rp_spec.cpuAllocation.shares.level = self.cpu_shares
    if self.cpu_allocation_shares and self.cpu_shares == 'custom':
        if self.cpu_allocation_shares != self.resource_pool_obj.config.cpuAllocation.shares.shares:
            changed = True
            rp_spec.cpuAllocation.shares.shares = self.cpu_allocation_shares
    if self.cpu_limit and self.cpu_limit != self.resource_pool_obj.config.cpuAllocation.limit:
        changed = True
        rp_spec.cpuAllocation.limit = self.cpu_limit
    if self.cpu_reservation and self.cpu_reservation != self.resource_pool_obj.config.cpuAllocation.reservation:
        changed = True
        rp_spec.cpuAllocation.reservation = self.cpu_reservation
    if self.cpu_expandable_reservations != self.resource_pool_obj.config.cpuAllocation.expandableReservation:
        changed = True
        rp_spec.cpuAllocation.expandableReservation = self.cpu_expandable_reservations
    if self.module.check_mode:
        self.module.exit_json(changed=changed)
    if changed:
        self.resource_pool_obj.UpdateConfig(self.resource_pool, rp_spec)
    resource_pool_config = self.generate_rp_config_return_value(True)
    self.module.exit_json(changed=changed, resource_pool_config=resource_pool_config)