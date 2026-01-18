import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_update_vlan(self, vlan):
    """
        Updates the properties of the given VLAN
        Only name and description are updated

        :param      vlan: The VLAN to update
        :type       vlan: :class:`NttCisetworkDomain`

        :return: an instance of `NttCisVlan`
        :rtype: :class:`NttCisVlan`
        """
    edit_node = ET.Element('editVlan', {'xmlns': TYPES_URN})
    edit_node.set('id', vlan.id)
    ET.SubElement(edit_node, 'name').text = vlan.name
    if vlan.description is not None:
        ET.SubElement(edit_node, 'description').text = vlan.description
    self.connection.request_with_orgId_api_2('network/editVlan', method='POST', data=ET.tostring(edit_node)).object
    return vlan