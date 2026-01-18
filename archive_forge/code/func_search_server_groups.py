import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def search_server_groups(self, name_or_id=None, filters=None):
    """Search server groups.

        :param name_or_id: Name or unique ID of the server group(s).
        :param filters: A dict containing additional filters to use.

        :returns: A list of compute ``ServerGroup`` objects matching the search
            criteria.
        :raises: :class:`~openstack.exceptions.SDKException` if something goes
            wrong during the OpenStack API call.
        """
    server_groups = self.list_server_groups()
    return _utils._filter_list(server_groups, name_or_id, filters)