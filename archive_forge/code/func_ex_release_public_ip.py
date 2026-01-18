import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_release_public_ip(self, address):
    """
        Release a public IP.

        :param address: CloudStackAddress which should be used
        :type  address: :class:`CloudStackAddress`

        :rtype: ``bool``
        """
    res = self._async_request(command='disassociateIpAddress', params={'id': address.id}, method='GET')
    return res['success']