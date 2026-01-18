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
def ex_get_control_access(self, node):
    """
        Returns the control access settings for specified node.

        :param  node: node to get the control access for
        :type   node: :class:`Node`

        :rtype: :class:`ControlAccess`
        """
    res = self.connection.request('%s/controlAccess' % get_url_path(node.id))
    everyone_access_level = None
    is_shared_elem = res.object.find(fixxpath(res.object, 'IsSharedToEveryone'))
    if is_shared_elem is not None and is_shared_elem.text == 'true':
        everyone_access_level = res.object.find(fixxpath(res.object, 'EveryoneAccessLevel')).text
    subjects = []
    xpath = fixxpath(res.object, 'AccessSettings/AccessSetting')
    for elem in res.object.findall(xpath):
        access_level = elem.find(fixxpath(res.object, 'AccessLevel')).text
        subject_elem = elem.find(fixxpath(res.object, 'Subject'))
        if subject_elem.get('type') == 'application/vnd.vmware.admin.group+xml':
            subj_type = 'group'
        else:
            subj_type = 'user'
        path = get_url_path(subject_elem.get('href'))
        res = self.connection.request(path)
        name = res.object.get('name')
        subject = Subject(type=subj_type, name=name, access_level=access_level, id=subject_elem.get('href'))
        subjects.append(subject)
    return ControlAccess(node, everyone_access_level, subjects)