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
def list_hypervisors(self, filters=None):
    """List all hypervisors

        :param filters:
        :returns: A list of compute ``Hypervisor`` objects.
        """
    if not filters:
        filters = {}
    return list(self.compute.hypervisors(details=True, **filters))