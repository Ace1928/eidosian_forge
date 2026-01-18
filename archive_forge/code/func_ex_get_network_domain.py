import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_get_network_domain(self, network_domain_id):
    """
        Get an individual Network Domain, by identifier

        :param      network_domain_id: The identifier of the network domain
        :type       network_domain_id: ``str``

        :rtype: :class:`NttCisNetworkDomain`
        """
    locations = self.list_locations()
    net = self.connection.request_with_orgId_api_2('network/networkDomain/%s' % network_domain_id).object
    return self._to_network_domain(net, locations)