import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_initiate_manual_snapshot(self, name=None, server_id=None):
    """
        Initiate a manual snapshot of server on the fly

        :param name: optional name of server
        :type name: ``str``

        :param server_id: optional parameter to use instead of name
        :type `server_id`str``

        :return: True of False
        :rtype: ``bool``

        """
    if server_id is None:
        node = self.list_nodes(ex_name=name)
        if len(node) > 1:
            raise RuntimeError('Found more than one server Id, please use one the following along with name parameter: {}'.format([n.id for n in node]))
    else:
        node = []
        node.append(self.ex_get_node_by_id(server_id))
    update_node = ET.Element('initiateManualSnapshot', {'xmlns': TYPES_URN})
    update_node.set('serverId', node[0].id)
    result = self.connection.request_with_orgId_api_2('snapshot/initiateManualSnapshot', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']