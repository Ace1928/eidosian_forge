import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_clean_failed_deployment(self, node):
    """
        Removes a node that has failed to deploy

        :param  node: The failed node to clean
        :type   node: :class:`Node` or ``str``
        """
    node_id = self._node_to_node_id(node)
    request_elm = ET.Element('cleanServer', {'xmlns': TYPES_URN, 'id': node_id})
    body = self.connection.request_with_orgId_api_2('server/cleanServer', method='POST', data=ET.tostring(request_elm)).object
    response_code = findtext(body, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']