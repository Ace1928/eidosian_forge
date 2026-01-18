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
def find_chassis(self, name_or_id, ignore_missing=True):
    """Find a single chassis.

        :param str name_or_id: The ID of a chassis.
        :param bool ignore_missing: When set to ``False``, an exception of
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the chassis does not exist.  When set to `True``, None will
            be returned when attempting to find a nonexistent chassis.
        :returns: One :class:`~openstack.baremetal.v1.chassis.Chassis` object
            or None.
        """
    return self._find(_chassis.Chassis, name_or_id, ignore_missing=ignore_missing)