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
def get_flavor_by_ram(self, ram, include=None, get_extra=True):
    """Get a flavor based on amount of RAM available.

        Finds the flavor with the least amount of RAM that is at least
        as much as the specified amount. If `include` is given, further
        filter based on matching flavor name.

        :param int ram: Minimum amount of RAM.
        :param string include: If given, will return a flavor whose name
            contains this string as a substring.
        :param get_extra:

        :returns: A compute ``Flavor`` object.
        :raises: :class:`~openstack.exceptions.SDKException` if no
            matching flavour could be found.
        """
    flavors = self.list_flavors(get_extra=get_extra)
    for flavor in sorted(flavors, key=operator.itemgetter('ram')):
        if flavor['ram'] >= ram and (not include or include in flavor['name']):
            return flavor
    raise exceptions.SDKException("Could not find a flavor with {ram} and '{include}'".format(ram=ram, include=include))