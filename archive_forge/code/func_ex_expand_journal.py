import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_expand_journal(self, consistency_group_id, size_gb):
    """
        Expand the consistency group's journhal size in 100Gb increments.

        :param consistency_group_id: The consistency group's UUID
        :type  consistency_group_id: ``str``

        :param size_gb: Gb in 100 Gb increments
        :type  size_gb: ``str``

        :return: True if response_code contains either 'IN_PROGRESS' or 'OK'
            otherwise False
        :rtype: ``bool``
        """
    expand_elm = ET.Element('expandJournal', {'id': consistency_group_id, 'xmlns': TYPES_URN})
    ET.SubElement(expand_elm, 'sizeGb').text = size_gb
    response = self.connection.request_with_orgId_api_2('consistencyGroup/expandJournal', method='POST', data=ET.tostring(expand_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']