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
def delete_chassis(self, chassis, ignore_missing=True):
    """Delete a chassis.

        :param chassis: The value can be either the ID of a chassis or
            a :class:`~openstack.baremetal.v1.chassis.Chassis` instance.
        :param bool ignore_missing: When set to ``False``, an exception
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the chassis could not be found. When set to ``True``, no
            exception will be raised when attempting to delete a non-existent
            chassis.

        :returns: The instance of the chassis which was deleted.
        :rtype: :class:`~openstack.baremetal.v1.chassis.Chassis`.
        """
    return self._delete(_chassis.Chassis, chassis, ignore_missing=ignore_missing)