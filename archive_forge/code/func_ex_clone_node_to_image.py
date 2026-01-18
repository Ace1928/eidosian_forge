import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_clone_node_to_image(self, node, image_name, image_description=None, cluster_id=None, is_guest_Os_Customization=None, tag_key_id=None, tag_value=None):
    """
        Clone a server into a customer image.

        :param  node: The server to clone
        :type   node: :class:`Node`

        :param  image_name: The name of the clone image
        :type   image_name: ``str``

        :param  description: The description of the image
        :type   description: ``str``

        :rtype: ``bool``
        """
    if image_description is None:
        image_description = ''
    node_id = self._node_to_node_id(node)
    "\n        Removing anything below 2.4\n        # Version 2.3 and lower\n        if LooseVersion(self.connection.active_api_version) < LooseVersion(\n                '2.4'):\n            response = self.connection.request_with_orgId_api_1(\n                'server/%s?clone=%s&desc=%s' %\n                (node_id, image_name, image_description)).object\n        # Version 2.4 and higher\n        else:\n        "
    clone_server_elem = ET.Element('cloneServer', {'id': node_id, 'xmlns': TYPES_URN})
    ET.SubElement(clone_server_elem, 'imageName').text = image_name
    if image_description is not None:
        ET.SubElement(clone_server_elem, 'description').text = image_description
    if cluster_id is not None:
        ET.SubElement(clone_server_elem, 'clusterId').text = cluster_id
    if is_guest_Os_Customization is not None:
        ET.SubElement(clone_server_elem, 'guestOsCustomization').text = is_guest_Os_Customization
    if tag_key_id is not None:
        tag_elem = ET.SubElement(clone_server_elem, 'tagById')
        ET.SubElement(tag_elem, 'tagKeyId').text = tag_key_id
        if tag_value is not None:
            ET.SubElement(tag_elem, 'value').text = tag_value
    response = self.connection.request_with_orgId_api_2('server/cloneServer', method='POST', data=ET.tostring(clone_server_elem)).object
    "\n        removing references to anything lower than 2.4\n        # Version 2.3 and lower\n        if LooseVersion(self.connection.active_api_version) < LooseVersion(\n                '2.4'):\n            response_code = findtext(response, 'result', GENERAL_NS)\n        else:\n        "
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'SUCCESS']