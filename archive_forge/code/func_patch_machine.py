import contextlib
import sys
import warnings
import jsonpatch
from openstack.baremetal.v1._proxy import Proxy
from openstack import exceptions
from openstack import warnings as os_warnings
def patch_machine(self, name_or_id, patch):
    """Patch Machine Information

        This method allows for an interface to manipulate node entries
        within Ironic.

        :param string name_or_id: A machine name or UUID to be updated.
        :param patch:
            The JSON Patch document is a list of dictonary objects that comply
            with RFC 6902 which can be found at
            https://tools.ietf.org/html/rfc6902.

            Example patch construction::

                patch=[]
                patch.append({
                    'op': 'remove',
                    'path': '/instance_info'
                })
                patch.append({
                    'op': 'replace',
                    'path': '/name',
                    'value': 'newname'
                })
                patch.append({
                    'op': 'add',
                    'path': '/driver_info/username',
                    'value': 'administrator'
                })

        :returns: Current state of the node.
        :rtype: :class:`~openstack.baremetal.v1.node.Node`.
        :raises: :class:`~openstack.exceptions.SDKException` on operation
            error.
        """
    return self.baremetal.patch_node(name_or_id, patch)