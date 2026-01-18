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
def delete_allocation(self, allocation, ignore_missing=True):
    """Delete an allocation.

        :param allocation: The value can be the name or ID of an allocation or
            a :class:`~openstack.baremetal.v1.allocation.Allocation` instance.
        :param bool ignore_missing: When set to ``False``, an exception
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the allocation could not be found. When set to ``True``, no
            exception will be raised when attempting to delete a non-existent
            allocation.

        :returns: The instance of the allocation which was deleted.
        :rtype: :class:`~openstack.baremetal.v1.allocation.Allocation`.
        """
    return self._delete(_allocation.Allocation, allocation, ignore_missing=ignore_missing)