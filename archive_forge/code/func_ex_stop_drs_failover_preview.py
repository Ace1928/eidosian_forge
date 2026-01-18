import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_stop_drs_failover_preview(self, consistency_group_id):
    """
        Takes a Consistency Group out of PREVIEW_MODE and back to DRS_MODE

        :param consistency_group_id: Consistency Group's Id
        :type ``str``

        :return: True if response_code contains either 'IN_PROGRESS' or 'OK'
         otherwise False
        :rtype: ``bool``
        """
    preview_elm = ET.Element('stopPreviewSnapshot', {'consistencyGroupId': consistency_group_id, 'xmlns': TYPES_URN})
    response = self.connection.request_with_orgId_api_2('consistencyGroup/stopPreviewSnapshot', method='POST', data=ET.tostring(preview_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']