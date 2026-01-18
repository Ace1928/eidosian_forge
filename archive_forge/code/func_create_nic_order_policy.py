from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def create_nic_order_policy(self):
    """
        Create a NIC order Policy
        Returns: NIC order Policy object
        """
    for active_nic in self.teaming_failover_order_active:
        if active_nic not in self.switch_object.spec.bridge.nicDevice:
            self.module.fail_json(msg="NIC '%s' (active) is not configured on vSwitch '%s'" % (active_nic, self.switch))
    for standby_nic in self.teaming_failover_order_standby:
        if standby_nic not in self.switch_object.spec.bridge.nicDevice:
            self.module.fail_json(msg="NIC '%s' (standby) is not configured on vSwitch '%s'" % (standby_nic, self.switch))
    nic_order = vim.host.NetworkPolicy.NicOrderPolicy()
    nic_order.activeNic = self.teaming_failover_order_active
    nic_order.standbyNic = self.teaming_failover_order_standby
    return nic_order