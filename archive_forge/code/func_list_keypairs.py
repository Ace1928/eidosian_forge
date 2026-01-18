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
def list_keypairs(self, filters=None):
    """List all available keypairs.

        :param filters:
        :returns: A list of compute ``Keypair`` objects.
        """
    if not filters:
        filters = {}
    return list(self.compute.keypairs(**filters))