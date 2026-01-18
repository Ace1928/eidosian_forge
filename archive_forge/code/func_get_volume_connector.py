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
def get_volume_connector(self, volume_connector, fields=None):
    """Get a specific volume_connector.

        :param volume_connector: The value can be the ID of a
            volume_connector or a
            :class:`~openstack.baremetal.v1.volume_connector.VolumeConnector`
            instance.
        :param fields: Limit the resource fields to fetch.`

        :returns: One
            :class: `~openstack.baremetal.v1.volume_connector.VolumeConnector`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            volume_connector matching the name or ID could be found.`
        """
    return self._get_with_fields(_volumeconnector.VolumeConnector, volume_connector, fields=fields)