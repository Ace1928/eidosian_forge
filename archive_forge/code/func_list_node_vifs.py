from openstack.baremetal.v1 import _common
from openstack.baremetal.v1 import allocation as _allocation
from openstack.baremetal.v1 import chassis as _chassis
from openstack.baremetal.v1 import conductor as _conductor
from openstack.baremetal.v1 import deploy_templates as _deploytemplates
from openstack.baremetal.v1 import driver as _driver
from openstack.baremetal.v1 import node as _node
from openstack.baremetal.v1 import port as _port
from openstack.baremetal.v1 import port_group as _portgroup
from openstack.baremetal.v1 import volume_connector as _volumeconnector
from openstack.baremetal.v1 import volume_target as _volumetarget
from openstack import exceptions
from openstack import proxy
from openstack import utils
def list_node_vifs(self, node):
    """List IDs of VIFs attached to the node.

        The exact form of the VIF ID depends on the network interface used by
        the node. In the most common case it is a Network service port
        (NOT a Bare Metal port) ID.

        :param node: The value can be either the name or ID of a node or
            a :class:`~openstack.baremetal.v1.node.Node` instance.
        :return: List of VIF IDs as strings.
        :raises: :exc:`~openstack.exceptions.NotSupported` if the server
            does not support the VIF API.
        """
    res = self._get_resource(_node.Node, node)
    return res.list_vifs(self)