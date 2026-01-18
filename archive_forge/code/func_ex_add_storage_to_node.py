import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_add_storage_to_node(self, amount, node=None, speed='STANDARD', controller_id=None, scsi_id=None):
    """
        Updated 8/23/18
        Add storage to the node
        One of node or controller_id must be selected

        :param  node: The server to add storage to (required if
                      controller_id is not used
        :type   node: :class:`Node`

        :param  amount: The amount of storage to add, in GB
        :type   amount: ``int``

        :param  speed: The disk speed type
        :type   speed: ``str``

        :param  conrollter_id: The disk may be added using the
                               cotnroller id (required if node
                               object is not used)
        :type   controller_id: ``str``

        :param  scsi_id: The target SCSI ID (optional)
        :type   scsi_id: ``int``

        :rtype: ``bool``
        """
    if node is None and controller_id is None or (node is not None and controller_id is not None):
        raise RuntimeError('Either a node or a controller id must be specified')
    update_node = ET.Element('addDisk', {'xmlns': TYPES_URN})
    if node is not None:
        ET.SubElement(update_node, 'serverId').text = node.id
    elif controller_id is not None:
        scsi_node = ET.Element('scsiController')
        ET.SubElement(scsi_node, 'controllerId').text = controller_id
        update_node.insert(1, scsi_node)
    ET.SubElement(update_node, 'sizeGb').text = str(amount)
    ET.SubElement(update_node, 'speed').text = speed.upper()
    if scsi_id is not None:
        ET.SubElement(update_node, 'scsiId').text = str(scsi_id)
    result = self.connection.request_with_orgId_api_2('server/addDisk', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']