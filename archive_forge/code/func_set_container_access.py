import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def set_container_access(self, name, access, refresh=False):
    """Set the access control list on a container.

        :param str name: Name of the container.
        :param str access: ACL string to set on the container. Can also be
            ``public`` or ``private`` which will be translated into appropriate
            ACL strings.
        :param refresh: Flag to trigger refresh of the container properties
        """
    if access not in OBJECT_CONTAINER_ACLS:
        raise exceptions.SDKException('Invalid container access specified: %s.  Must be one of %s' % (access, list(OBJECT_CONTAINER_ACLS.keys())))
    return self.object_store.set_container_metadata(name, read_ACL=OBJECT_CONTAINER_ACLS[access], refresh=refresh)