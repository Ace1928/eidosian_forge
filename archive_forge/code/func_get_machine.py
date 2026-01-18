import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def get_machine(self, name_or_id):
    """Get Machine by name or uuid

        Search the baremetal host out by utilizing the supplied id value
        which can consist of a name or UUID.

        :param name_or_id: A node name or UUID that will be looked up.

        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :returns: The node found or None if no nodes are found.
        """
    try:
        return self.baremetal.find_node(name_or_id, ignore_missing=False)
    except exceptions.NotFoundException:
        return None