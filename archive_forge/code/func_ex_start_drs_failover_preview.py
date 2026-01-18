import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_start_drs_failover_preview(self, consistency_group_id, snapshot_id):
    """
        Brings a Consistency Group into PREVIEWING_SNAPSHOT mode.

        :param consistency_group_id: Id of the Consistency Group to put into
                                     PRIEVEW_MODE
        :type consistency_group_id: ``str``

        :param snapshot_id: Id of the Snapshot to preview
        :type snapshot_id: ``str``

        :return: True if response_code contains either 'IN_PROGRESS' or 'OK'
            otherwise False
        :rtype: ``bool``
        """
    preview_elm = ET.Element('startPreviewSnapshot', {'consistencyGroupId': consistency_group_id, 'xmlns': TYPES_URN})
    ET.SubElement(preview_elm, 'snapshotId').text = snapshot_id
    response = self.connection.request_with_orgId_api_2('consistencyGroup/startPreviewSnapshot', method='POST', data=ET.tostring(preview_elm)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']