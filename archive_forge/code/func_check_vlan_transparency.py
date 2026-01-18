import abc
from neutron_lib.api.definitions import portbindings
def check_vlan_transparency(self, context):
    """Check if the network supports vlan transparency.

        :param context: NetworkContext instance describing the network.

        Check if the network supports vlan transparency or not.
        """
    pass