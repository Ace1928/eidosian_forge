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
def patch_port(self, port, patch):
    """Apply a JSON patch to the port.

        :param port: The value can be the ID of a port or a
            :class:`~openstack.baremetal.v1.port.Port` instance.
        :param patch: JSON patch to apply.

        :returns: The updated port.
        :rtype: :class:`~openstack.baremetal.v1.port.Port`
        """
    return self._get_resource(_port.Port, port).patch(self, patch)