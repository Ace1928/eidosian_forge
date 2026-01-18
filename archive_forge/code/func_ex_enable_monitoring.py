import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_enable_monitoring(self, node, service_plan='ESSENTIALS'):
    """
        Enables cloud monitoring on a node

        :param   node: The node to monitor
        :type    node: :class:`Node`

        :param   service_plan: The service plan, one of ESSENTIALS or
                               ADVANCED
        :type    service_plan: ``str``

        :rtype: ``bool``
        """
    update_node = ET.Element('enableServerMonitoring', {'xmlns': TYPES_URN})
    update_node.set('id', node.id)
    ET.SubElement(update_node, 'servicePlan').text = service_plan
    result = self.connection.request_with_orgId_api_2('server/enableServerMonitoring', method='POST', data=ET.tostring(update_node)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']