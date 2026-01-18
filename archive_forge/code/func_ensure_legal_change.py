from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
def ensure_legal_change(self):
    """
        Ensure the change (if any) represented by the VlanDiff represents a legal change to VLAN state.

        - private_ipv4_base_address cannot be changed
        - private_ipv4_prefix_size must be greater than or equal to the VLAN's existing private_ipv4_range_size

        :raise InvalidVlanChangeError: The VlanDiff does not represent a legal change to VLAN state.
        """
    if self.private_ipv4_base_address_changed:
        raise InvalidVlanChangeError('Cannot change the private IPV4 base address for an existing VLAN.')
    if self.private_ipv4_prefix_size_increased:
        raise InvalidVlanChangeError('Cannot shrink the private IPV4 network for an existing VLAN (only expand is supported).')