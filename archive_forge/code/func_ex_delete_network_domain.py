import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_delete_network_domain(self, network_domain):
    """
        Delete a network domain

        :param      network_domain: The network domain to delete
        :type       network_domain: :class:`NttCisNetworkDomain`

        :rtype: ``bool``
        """
    delete_node = ET.Element('deleteNetworkDomain', {'xmlns': TYPES_URN})
    delete_node.set('id', network_domain.id)
    result = self.connection.request_with_orgId_api_2('network/deleteNetworkDomain', method='POST', data=ET.tostring(delete_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']