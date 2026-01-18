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
def ex_set_control_access(self, node, control_access):
    """
        Sets control access for the specified node.

        :param  node: node
        :type   node: :class:`Node`

        :param  control_access: control access settings
        :type   control_access: :class:`ControlAccess`

        :rtype: ``None``
        """
    xml = ET.Element('ControlAccessParams', {'xmlns': 'http://www.vmware.com/vcloud/v1.5'})
    shared_to_everyone = ET.SubElement(xml, 'IsSharedToEveryone')
    if control_access.everyone_access_level:
        shared_to_everyone.text = 'true'
        everyone_access_level = ET.SubElement(xml, 'EveryoneAccessLevel')
        everyone_access_level.text = control_access.everyone_access_level
    else:
        shared_to_everyone.text = 'false'
    if control_access.subjects:
        access_settings_elem = ET.SubElement(xml, 'AccessSettings')
    for subject in control_access.subjects:
        setting = ET.SubElement(access_settings_elem, 'AccessSetting')
        if subject.id:
            href = subject.id
        else:
            res = self.ex_query(type=subject.type, filter='name==' + subject.name)
            if not res:
                raise LibcloudError('Specified subject "{} {}" not found '.format(subject.type, subject.name))
            href = res[0]['href']
        ET.SubElement(setting, 'Subject', {'href': href})
        ET.SubElement(setting, 'AccessLevel').text = subject.access_level
    headers = {'Content-Type': 'application/vnd.vmware.vcloud.controlAccess+xml'}
    self.connection.request('%s/action/controlAccess' % get_url_path(node.id), data=ET.tostring(xml), headers=headers, method='POST')