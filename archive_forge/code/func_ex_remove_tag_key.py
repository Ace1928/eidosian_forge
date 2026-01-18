import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_remove_tag_key(self, tag_key):
    """
        Modify a specific tag key

        :param tag_key: The tag key you want to remove (required)
        :type  tag_key: :class:`NttCisTagKey` or ``str``

        :rtype: ``bool``
        """
    tag_key_id = self._tag_key_to_tag_key_id(tag_key)
    remove_tag_key = ET.Element('deleteTagKey', {'xmlns': TYPES_URN, 'id': tag_key_id})
    response = self.connection.request_with_orgId_api_2('tag/deleteTagKey', method='POST', data=ET.tostring(remove_tag_key)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']