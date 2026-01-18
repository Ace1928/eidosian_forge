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
def ex_find_vm_nodes(self, vm_name, max_results=50):
    """
        Finds nodes that contain a VM with the specified name.

        :param vm_name: The VM name to find nodes for
        :type vm_name: ``str``

        :param max_results: Maximum number of results up to 128
        :type max_results: ``int``

        :return: List of node instances
        :rtype: `list` of :class:`Node`
        """
    vms = self.ex_query('vm', filter='name=={vm_name}'.format(vm_name=vm_name), page=1, page_size=max_results)
    return [self._ex_get_node(vm['container']) for vm in vms]