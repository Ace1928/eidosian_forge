import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_modify_tag_key(self, tag_key, name=None, description=None, value_required=None, display_on_report=None):
    """
        Modify a specific tag key

        :param tag_key: The tag key you want to modify (required)
        :type  tag_key: :class:`NttCisTagKey` or ``str``

        :param name: Set to modify the name of the tag key
        :type  name: ``str``

        :param description: Set to modify the description of the tag key
        :type  description: ``str``

        :param value_required: Set to modify if a value is required for
                               the tag key
        :type  value_required: ``bool``

        :param display_on_report: Set to modify if this tag key should display
                                  on the usage reports
        :type  display_on_report: ``bool``

        :rtype: ``bool``
        """
    tag_key_id = self._tag_key_to_tag_key_id(tag_key)
    modify_tag_key = ET.Element('editTagKey', {'xmlns': TYPES_URN, 'id': tag_key_id})
    if name is not None:
        ET.SubElement(modify_tag_key, 'name').text = name
    if description is not None:
        ET.SubElement(modify_tag_key, 'description').text = description
    if value_required is not None:
        ET.SubElement(modify_tag_key, 'valueRequired').text = str(value_required).lower()
    if display_on_report is not None:
        ET.SubElement(modify_tag_key, 'displayOnReport').text = str(display_on_report).lower()
    response = self.connection.request_with_orgId_api_2('tag/editTagKey', method='POST', data=ET.tostring(modify_tag_key)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']