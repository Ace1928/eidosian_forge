import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_vpc_offerings(self):
    """
        List the available vpc offerings

        :rtype ``list`` of :class:`CloudStackVPCOffering`
        """
    res = self._sync_request(command='listVPCOfferings', method='GET')
    vpcoffers = res.get('vpcoffering', [])
    vpcofferings = []
    for vpcoffer in vpcoffers:
        vpcofferings.append(CloudStackVPCOffering(vpcoffer['name'], vpcoffer['displaytext'], vpcoffer['id'], self))
    return vpcofferings