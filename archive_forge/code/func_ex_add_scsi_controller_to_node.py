import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_add_scsi_controller_to_node(self, server_id, adapter_type, bus_number=None):
    """
        Added 8/27/18:  Adds a SCSI Controller by node id
        :param server_id: server id
        :param adapter_type: the type of SCSI Adapter, i.e., LSI_LOGIC_PARALLEL
        :param bus_number: optional number of server's bus
        :return: whether addition is in progress or 'OK' otherwise false
        """
    update_node = ET.Element('addScsiController', {'xmlns': TYPES_URN})
    ET.SubElement(update_node, 'serverId').text = server_id
    ET.SubElement(update_node, 'adapterType').text = adapter_type
    if bus_number is not None:
        ET.SubElement(update_node, 'busNumber').text = bus_number
    result = self.connection.request_with_orgId_api_2('server/addScsiController', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']