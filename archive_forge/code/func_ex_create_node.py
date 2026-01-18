from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_create_node(self, network_domain_id, name, ip, ex_description=None, connection_limit=25000, connection_rate_limit=2000):
    """
        Create a new node

        :param network_domain_id: Network Domain ID (required)
        :type  name: ``str``

        :param name: name of the node (required)
        :type  name: ``str``

        :param ip: IPv4 address of the node (required)
        :type  ip: ``str``

        :param ex_description: Description of the node (required)
        :type  ex_description: ``str``

        :param connection_limit: Maximum number
                of concurrent connections per sec
        :type  connection_limit: ``int``

        :param connection_rate_limit: Maximum number of concurrent sessions
        :type  connection_rate_limit: ``int``

        :return: Instance of ``NttCisVIPNode``
        :rtype: ``NttCisVIPNode``
        """
    create_node_elm = ET.Element('createNode', {'xmlns': TYPES_URN})
    ET.SubElement(create_node_elm, 'networkDomainId').text = network_domain_id
    ET.SubElement(create_node_elm, 'name').text = name
    if ex_description is not None:
        ET.SubElement(create_node_elm, 'description').text = str(ex_description)
    ET.SubElement(create_node_elm, 'ipv4Address').text = ip
    ET.SubElement(create_node_elm, 'status').text = 'ENABLED'
    ET.SubElement(create_node_elm, 'connectionLimit').text = str(connection_limit)
    ET.SubElement(create_node_elm, 'connectionRateLimit').text = str(connection_rate_limit)
    response = self.connection.request_with_orgId_api_2(action='networkDomainVip/createNode', method='POST', data=ET.tostring(create_node_elm)).object
    node_id = None
    node_name = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'nodeId':
            node_id = info.get('value')
        if info.get('name') == 'name':
            node_name = info.get('value')
    return NttCisVIPNode(id=node_id, name=node_name, status=State.RUNNING, ip=ip)