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
def ex_find_node(self, node_name, vdcs=None):
    """
        Searches for node across specified vDCs. This is more effective than
        querying all nodes to get a single instance.

        :param node_name: The name of the node to search for
        :type node_name: ``str``

        :param vdcs: None, vDC or a list of vDCs to search in. If None all vDCs
                     will be searched.
        :type vdcs: :class:`Vdc`

        :return: node instance or None if not found
        :rtype: :class:`Node` or ``None``
        """
    if not vdcs:
        vdcs = self.vdcs
    if not getattr(vdcs, '__iter__', False):
        vdcs = [vdcs]
    for vdc in vdcs:
        res = self.connection.request(get_url_path(vdc.id))
        xpath = fixxpath(res.object, 'ResourceEntities/ResourceEntity')
        entity_elems = res.object.findall(xpath)
        for entity_elem in entity_elems:
            if entity_elem.get('type') == 'application/vnd.vmware.vcloud.vApp+xml' and entity_elem.get('name') == node_name:
                path = entity_elem.get('href')
                return self._ex_get_node(path)
    return None