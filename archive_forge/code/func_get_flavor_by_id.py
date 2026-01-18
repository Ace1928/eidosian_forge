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
def get_flavor_by_id(self, id, get_extra=False):
    """Get a flavor by ID

        :param id: ID of the flavor.
        :param get_extra: Whether or not the list_flavors call should get the
            extra flavor specs.
        :returns: A compute ``Flavor`` object if found, else None.
        """
    return self.compute.get_flavor(id, get_extra_specs=get_extra)