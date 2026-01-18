import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_update_network_domain(self, network_domain):
    """
        Update the properties of a network domain

        :param      network_domain: The network domain with updated properties
        :type       network_domain: :class:`NttCisNetworkDomain`

        :return: an instance of `NttCisNetworkDomain`
        :rtype: :class:`NttCisNetworkDomain`
        """
    edit_node = ET.Element('editNetworkDomain', {'xmlns': TYPES_URN})
    edit_node.set('id', network_domain.id)
    ET.SubElement(edit_node, 'name').text = network_domain.name
    if network_domain.description is not None:
        ET.SubElement(edit_node, 'description').text = network_domain.description
    ET.SubElement(edit_node, 'type').text = network_domain.plan
    self.connection.request_with_orgId_api_2('network/editNetworkDomain', method='POST', data=ET.tostring(edit_node)).object
    return network_domain