import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_change_nic_network_adapter(self, nic_id, network_adapter_name):
    """
        Change network adapter of a NIC on a cloud server

        :param    nic_id:  Nic ID
        :type     nic_id: :``str``

        :param    network_adapter_name:  Network adapter name
        :type     network_adapter_name: :``str``

        :rtype: ``bool``
        """
    change_elem = ET.Element('changeNetworkAdapter', {'nicId': nic_id, 'xmlns': TYPES_URN})
    ET.SubElement(change_elem, 'networkAdapter').text = network_adapter_name
    response = self.connection.request_with_orgId_api_2('server/changeNetworkAdapter', method='POST', data=ET.tostring(change_elem)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']