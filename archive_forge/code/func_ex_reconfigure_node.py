import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_reconfigure_node(self, node, memory_gb=None, cpu_count=None, cores_per_socket=None, cpu_performance=None):
    """
        Reconfigure the virtual hardware specification of a node

        :param  node: The server to change
        :type   node: :class:`Node`

        :param  memory_gb: The amount of memory in GB (optional)
        :type   memory_gb: ``int``

        :param  cpu_count: The number of CPU (optional)
        :type   cpu_count: ``int``

        :param  cores_per_socket: Number of CPU cores per socket (optional)
        :type   cores_per_socket: ``int``

        :param  cpu_performance: CPU Performance type (optional)
        :type   cpu_performance: ``str``

        :rtype: ``bool``
        """
    update = ET.Element('reconfigureServer', {'xmlns': TYPES_URN})
    update.set('id', node.id)
    if memory_gb is not None:
        ET.SubElement(update, 'memoryGb').text = str(memory_gb)
    if cpu_count is not None:
        ET.SubElement(update, 'cpuCount').text = str(cpu_count)
    if cpu_performance is not None:
        ET.SubElement(update, 'cpuSpeed').text = cpu_performance
    if cores_per_socket is not None:
        ET.SubElement(update, 'coresPerSocket').text = str(cores_per_socket)
    result = self.connection.request_with_orgId_api_2('server/reconfigureServer', method='POST', data=ET.tostring(update)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']