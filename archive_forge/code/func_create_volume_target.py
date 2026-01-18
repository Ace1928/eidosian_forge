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
def create_volume_target(self, **attrs):
    """Create a new volume_target from attributes.

        :param dict attrs: Keyword arguments that will be used to create a
            :class:`~openstack.baremetal.v1.volume_target.VolumeTarget`.

        :returns: The results of volume_target creation.
        :rtype:
            :class:`~openstack.baremetal.v1.volume_target.VolumeTarget`.
        """
    return self._create(_volumetarget.VolumeTarget, **attrs)