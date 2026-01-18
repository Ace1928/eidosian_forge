from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_create_virtual_listener(self, network_domain_id, name, ex_description, port=None, pool=None, listener_ip_address=None, persistence_profile=None, fallback_persistence_profile=None, irule=None, protocol='TCP', optimization_profile='TCP', connection_limit=25000, connection_rate_limit=2000, source_port_preservation='PRESERVE'):
    """
        Create a new virtual listener (load balancer)

        :param network_domain_id: Network Domain ID (required)
        :type  name: ``str``

        :param name: name of the listener (required)
        :type  name: ``str``

        :param ex_description: Description of the node (required)
        :type  ex_description: ``str``

        :param port: An integer in the range of 1-65535. If not supplied,
                     it will be taken to mean 'Any Port'
        :type  port: ``int``

        :param pool: The pool to use for the listener
        :type  pool: :class:`NttCisPool`

        :param listener_ip_address: The IPv4 Address of the virtual listener
        :type  listener_ip_address: ``str``

        :param persistence_profile: Persistence profile
        :type  persistence_profile: :class:`NttCisPersistenceProfile`

        :param fallback_persistence_profile: Fallback persistence profile
        :type  fallback_persistence_profile:
            :class:`NttCisPersistenceProfile`

        :param irule: The iRule to apply
        :type  irule: :class:`NttCisDefaultiRule`

        :param protocol: For STANDARD type, ANY, TCP or UDP
                         for PERFORMANCE_LAYER_4 choice of ANY, TCP, UDP, HTTP
        :type  protocol: ``str``

        :param optimization_profile: For STANDARD type and protocol
                                     TCP an optimization type of TCP,
                                     LAN_OPT, WAN_OPT, MOBILE_OPT,
                                     or TCP_LEGACY is required.
                                     Default is 'TCP'.
        :type  protocol: ``str``

        :param connection_limit: Maximum number
                                of concurrent connections per sec
        :type  connection_limit: ``int``

        :param connection_rate_limit: Maximum number of concurrent sessions
        :type  connection_rate_limit: ``int``

        :param source_port_preservation: Choice of PRESERVE,
                                         PRESERVE_STRICT or CHANGE
        :type  source_port_preservation: ``str``

        :return: Instance of the listener
        :rtype: ``NttCisVirtualListener``
        """
    if port == 80 or port == 443:
        listener_type = 'PERFORMANCE_LAYER_4'
    else:
        listener_type = 'STANDARD'
    if listener_type == 'STANDARD' and optimization_profile is None:
        raise ValueError(' CONFIGURATION_NOT_SUPPORTED: optimizationProfile is required for type STANDARD and protocol TCP')
    create_node_elm = ET.Element('createVirtualListener', {'xmlns': TYPES_URN})
    ET.SubElement(create_node_elm, 'networkDomainId').text = network_domain_id
    ET.SubElement(create_node_elm, 'name').text = name
    ET.SubElement(create_node_elm, 'description').text = str(ex_description)
    ET.SubElement(create_node_elm, 'type').text = listener_type
    ET.SubElement(create_node_elm, 'protocol').text = protocol
    if listener_ip_address is not None:
        ET.SubElement(create_node_elm, 'listenerIpAddress').text = str(listener_ip_address)
    if port is not None:
        ET.SubElement(create_node_elm, 'port').text = str(port)
    ET.SubElement(create_node_elm, 'enabled').text = 'true'
    ET.SubElement(create_node_elm, 'connectionLimit').text = str(connection_limit)
    ET.SubElement(create_node_elm, 'connectionRateLimit').text = str(connection_rate_limit)
    ET.SubElement(create_node_elm, 'sourcePortPreservation').text = source_port_preservation
    if pool is not None:
        ET.SubElement(create_node_elm, 'poolId').text = pool.id
    if persistence_profile is not None:
        ET.SubElement(create_node_elm, 'persistenceProfileId').text = persistence_profile.id
    if optimization_profile is not None:
        ET.SubElement(create_node_elm, 'optimizationProfile').text = optimization_profile
    if fallback_persistence_profile is not None:
        ET.SubElement(create_node_elm, 'fallbackPersistenceProfileId').text = fallback_persistence_profile.id
    if irule is not None:
        ET.SubElement(create_node_elm, 'iruleId').text = irule.id
    response = self.connection.request_with_orgId_api_2(action='networkDomainVip/createVirtualListener', method='POST', data=ET.tostring(create_node_elm)).object
    virtual_listener_id = None
    virtual_listener_ip = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'virtualListenerId':
            virtual_listener_id = info.get('value')
        if info.get('name') == 'listenerIpAddress':
            virtual_listener_ip = info.get('value')
    return NttCisVirtualListener(id=virtual_listener_id, name=name, ip=virtual_listener_ip, status=State.RUNNING)