import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_network_offerings(self):
    """
        List the available network offerings

        :rtype ``list`` of :class:`CloudStackNetworkOffering`
        """
    res = self._sync_request(command='listNetworkOfferings', method='GET')
    netoffers = res.get('networkoffering', [])
    networkofferings = []
    for netoffer in netoffers:
        networkofferings.append(CloudStackNetworkOffering(netoffer['name'], netoffer['displaytext'], netoffer['guestiptype'], netoffer['id'], netoffer['serviceofferingid'], netoffer['forvpc'], self))
    return networkofferings