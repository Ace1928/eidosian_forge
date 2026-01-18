import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_snapshot_preview_server(self, snapshot_id, server_name, server_started, nics_connected, server_description=None, target_cluster_id=None, preserve_mac_addresses=None, tag_key_name=None, tag_key_id=None, tag_value=None):
    """
        Create a snapshot preview of a server to clone to a new server

        :param snapshot_id: ID of the specific snahpshot to use in
                                 creating preview server.
        :type  snapshot_id: ``str``

        :param server_name: Name of the server created from the snapshot
        :type  ``str``

        :param nics_connected: 'true' or 'false'.  Should the nics be
                                automatically connected
        :type  ``str``

        :param server_description: (Optional) A brief description of the
                                   server.
        :type ``str``

        :param target_cluster_id: (Optional) The ID of a specific cluster as
                                   opposed to the default.
        :type ``str``

        :param preserve_mac_address: (Optional) If set to 'true' will preserve
                                      mac address from the original server.
        :type ``str``

        :param tag_key_name: (Optional) If tagging is desired and by name is
                             desired, set this to the tag name.
        :type ``str``

        :param tag_key_id: (Optional) If tagging is desired and by id is
                            desired, set this to the tag id.
        :type ``str``

        :param tag_value: (Optional) If using a tag_key_id or tag_key_name,
                           set the value fo tag_value.

        :rtype: ``str``
        """
    create_preview = ET.Element('createSnapshotPreviewServer', {'xmlns': TYPES_URN, 'snapshotId': snapshot_id})
    ET.SubElement(create_preview, 'serverName').text = server_name
    if server_description is not None:
        ET.SubElement(create_preview, 'serverDescription').text = server_description
    if target_cluster_id is not None:
        ET.SubElement(create_preview, 'targetClusterId').text = target_cluster_id
    ET.SubElement(create_preview, 'serverStarted').text = server_started
    ET.SubElement(create_preview, 'nicsConnected').text = nics_connected
    if preserve_mac_addresses is not None:
        ET.SubElement(create_preview, 'preserveMacAddresses').text = preserve_mac_addresses
    if tag_key_name is not None:
        tag_elem = ET.SubElement(create_preview, 'tag')
        ET.SubElement(tag_elem, 'tagKeyName').text = tag_key_name
        ET.SubElement(tag_elem, 'value').text = tag_value
    elif tag_key_id is not None:
        tag_elem = ET.SubElement(create_preview, 'tagById')
        ET.SubElement(create_preview, 'tagKeyId').text = tag_key_name
        ET.SubElement(tag_elem, 'value').text = tag_value
    result = self.connection.request_with_orgId_api_2('snapshot/createSnapshotPreviewServer', method='POST', data=ET.tostring(create_preview)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']