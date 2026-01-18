import os
import re
import copy
import time
import base64
import datetime
from xml.parsers.expat import ExpatError
from libcloud.utils.py3 import ET, b, next, httplib, urlparse, urlencode
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def ex_set_metadata_entry(self, node, key, value):
    """
        :param  node: node
        :type   node: :class:`Node`

        :param key: metadata key to be set
        :type key: ``str``

        :param value: metadata value to be set
        :type value: ``str``

        :rtype: ``None``
        """
    metadata_elem = ET.Element('Metadata', {'xmlns': 'http://www.vmware.com/vcloud/v1.5', 'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'})
    entry = ET.SubElement(metadata_elem, 'MetadataEntry')
    key_elem = ET.SubElement(entry, 'Key')
    key_elem.text = key
    value_elem = ET.SubElement(entry, 'Value')
    value_elem.text = value
    res = self.connection.request('%s/metadata' % get_url_path(node.id), data=ET.tostring(metadata_elem), headers={'Content-Type': 'application/vnd.vmware.vcloud.metadata+xml'}, method='POST')
    self._wait_for_task_completion(res.object.get('href'))