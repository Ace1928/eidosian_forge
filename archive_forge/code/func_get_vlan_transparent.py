from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api import validators
from neutron_lib import constants
def get_vlan_transparent(network):
    """Get the value of vlan_transparent from a network if set.

    :param network: The network dict to retrieve the value of vlan_transparent
        from.
    :returns: The value of vlan_transparent from the network dict if set in
        the dict, otherwise False is returned.
    """
    return network[VLANTRANSPARENT] if VLANTRANSPARENT in network and validators.is_attr_set(network[VLANTRANSPARENT]) else False