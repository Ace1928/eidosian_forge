from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def get_introspection_data(self, introspection, processed=True):
    """Get introspection data.

        :param introspection: The value can be the name or ID of an
            introspection (matching bare metal node name or ID) or
            an :class:`~.introspection.Introspection` instance.
        :param processed: Whether to fetch the final processed data (the
            default) or the raw unprocessed data as received from the ramdisk.
        :returns: introspection data from the most recent successful run.
        :rtype: dict
        """
    res = self._get_resource(_introspect.Introspection, introspection)
    return res.get_data(self, processed=processed)