import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def get_container_access(self, name):
    """Get the control list from a container.

        :param str name: Name of the container.
        :returns: The contol list for the container.
        :raises: :class:`~openstack.exceptions.SDKException` if the container
            was not found or container access could not be determined.
        """
    container = self.get_container(name, skip_cache=True)
    if not container:
        raise exceptions.SDKException('Container not found: %s' % name)
    acl = container.read_ACL or ''
    for key, value in OBJECT_CONTAINER_ACLS.items():
        if str(acl) == str(value):
            return key
    raise exceptions.SDKException('Could not determine container access for ACL: %s.' % acl)