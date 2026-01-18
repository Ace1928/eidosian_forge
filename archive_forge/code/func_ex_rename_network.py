import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_rename_network(self, network, new_name):
    """
        Rename a network in MCP 1 data center

        :param  network: The network to rename
        :type   network: :class:`NttCisNetwork`

        :param  new_name: The new name of the network
        :type   new_name: ``str``

        :rtype: ``bool``
        """
    response = self.connection.request_with_orgId_api_1('network/%s' % network.id, method='POST', data='name=%s' % new_name).object
    response_code = findtext(response, 'result', GENERAL_NS)
    return response_code == 'SUCCESS'