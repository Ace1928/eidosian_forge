import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_edit_metadata(self, node, name=None, description=None, drs_eligible=None):
    request_elem = ET.Element('editServerMetadata', {'xmlns': TYPES_URN, 'id': node.id})
    if name is not None:
        ET.SubElement(request_elem, 'name').text = name
    if description is not None:
        ET.SubElement(request_elem, 'description').text = description
    if drs_eligible is not None:
        ET.SubElement(request_elem, 'drsEligible').text = drs_eligible
    body = self.connection.request_with_orgId_api_2('server/editServerMetadata', method='POST', data=ET.tostring(request_elem)).object
    response_code = findtext(body, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']