import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_remove_storage(self, disk_id):
    """
        Remove storage from a node

        :param  node: The server to add storage to
        :type   node: :class:`Node`

        :param  disk_id: The ID of the disk to remove
        :type   disk_id: ``str``

        :rtype: ``bool``
        """
    remove_disk = ET.Element('removeDisk', {'xmlns': TYPES_URN})
    remove_disk.set('id', disk_id)
    result = self.connection.request_with_orgId_api_2('server/removeDisk', method='POST', data=ET.tostring(remove_disk)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']