import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_list_network_acllists(self):
    """
        Lists all network ACLs

        :rtype: ``list`` of :class:`CloudStackNetworkACLList`
        """
    acllists = []
    result = self._sync_request(command='listNetworkACLLists', method='GET')
    if not result:
        return acllists
    for acllist in result['networkacllist']:
        acllists.append(CloudStackNetworkACLList(acllist['id'], acllist['name'], acllist.get('vpcid', []), self, acllist['description']))
    return acllists