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
def search_flavors(self, name_or_id=None, filters=None, get_extra=True):
    """Search flavors.

        :param name_or_id:
        :param flavors:
        :param get_extra:
        :returns: A list of compute ``Flavor`` objects matching the search
            criteria.
        """
    flavors = self.list_flavors(get_extra=get_extra)
    return _utils._filter_list(flavors, name_or_id, filters)