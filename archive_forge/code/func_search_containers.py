import concurrent.futures
import urllib.parse
import keystoneauth1.exceptions
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1._proxy import Proxy
def search_containers(self, name=None, filters=None):
    """Search containers.

        :param string name: Container name.
        :param filters: A dict containing additional filters to use.
            OR
            A string containing a jmespath expression for further filtering.
            Example:: "[?last_name==`Smith`] | [?other.gender]==`Female`]"

        :returns: A list of object store ``Container`` objects matching the
            search criteria.
        :raises: :class:`~openstack.exceptions.SDKException`: If something goes
            wrong during the OpenStack API call.
        """
    containers = self.list_containers()
    return _utils._filter_list(containers, name, filters)