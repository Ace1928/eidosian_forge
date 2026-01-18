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
def list_flavors(self, get_extra=False):
    """List all available flavors.

        :param get_extra: Whether or not to fetch extra specs for each flavor.
            Defaults to True. Default behavior value can be overridden in
            clouds.yaml by setting openstack.cloud.get_extra_specs to False.
        :returns: A list of compute ``Flavor`` objects.
        """
    return list(self.compute.flavors(details=True, get_extra_specs=get_extra))