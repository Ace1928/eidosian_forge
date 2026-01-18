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
def set_node_boot_device(self, node, boot_device, persistent=False):
    """Set node boot device

        :param node: The value can be the name or ID of a node or a
            :class:`~openstack.baremetal.v1.node.Node` instance.
        :param boot_device: Boot device to assign to the node.
        :param persistent: If the boot device change is maintained after node
            reboot
        :return: The updated :class:`~openstack.baremetal.v1.node.Node`
        """
    res = self._get_resource(_node.Node, node)
    return res.set_boot_device(self, boot_device, persistent=persistent)