from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
from ansible.module_utils._text import to_native
def create_network_policy(self):
    """
        Create a Network Policy
        Returns: Network Policy object
        """
    security_policy = None
    shaping_policy = None
    teaming_policy = None
    if not all((option is None for option in [self.sec_promiscuous_mode, self.sec_mac_changes, self.sec_forged_transmits])):
        security_policy = self.create_security_policy()
    if self.ts_enabled:
        shaping_policy = self.create_shaping_policy()
    teaming_policy = self.create_teaming_policy()
    network_policy = vim.host.NetworkPolicy(security=security_policy, nicTeaming=teaming_policy, shapingPolicy=shaping_policy)
    return network_policy