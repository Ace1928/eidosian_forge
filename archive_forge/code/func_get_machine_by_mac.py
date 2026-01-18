import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def get_machine_by_mac(self, mac):
    """Get machine by port MAC address

        :param mac: Port MAC address to query in order to return a node.

        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :returns: The node found or None if no nodes are found.
        """
    nic = self.get_nic_by_mac(mac)
    if nic is None:
        return None
    else:
        return self.get_machine(nic['node_uuid'])