import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_destroy_nic(self, nic_id):
    """
        Remove a NIC on a node, removing the node from a VLAN

        :param      nic_id: The identifier of the NIC to remove
        :type       nic_id: ``str``

        :rtype: ``bool``
        """
    request = ET.Element('removeNic', {'xmlns': TYPES_URN, 'id': nic_id})
    response = self.connection.request_with_orgId_api_2('server/removeNic', method='POST', data=ET.tostring(request)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']