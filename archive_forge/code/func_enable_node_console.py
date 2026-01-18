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
def enable_node_console(self, node):
    """Enable the console for a node.

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :returns: None
        """
    res = self._get_resource(_node.Node, node)
    return res.set_console_mode(self, True)