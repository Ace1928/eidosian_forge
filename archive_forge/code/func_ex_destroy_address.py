import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_destroy_address(self, address):
    """
        Destroy a static address.

        :param  address: Address object to destroy
        :type   address: ``str`` or :class:`GCEAddress`

        :return:  True if successful
        :rtype:   ``bool``
        """
    if not hasattr(address, 'name'):
        address = self.ex_get_address(address)
    if hasattr(address.region, 'name'):
        request = '/regions/{}/addresses/{}'.format(address.region.name, address.name)
    else:
        request = '/global/addresses/%s' % address.name
    self.connection.async_request(request, method='DELETE')
    return True