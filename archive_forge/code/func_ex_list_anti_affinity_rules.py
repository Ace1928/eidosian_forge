import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_anti_affinity_rules(self, network=None, network_domain=None, node=None, filter_id=None, filter_state=None):
    """
        List anti affinity rules for a network, network domain, or node

        :param network: The network to list anti affinity rules for
                        One of network, network_domain, or node is required
        :type  network: :class:`NttCisNetwork` or ``str``

        :param network_domain: The network domain to list anti affinity rules
                               One of network, network_domain,
                               or node is required
        :type  network_domain: :class:`NttCisNetworkDomain` or ``str``

        :param node: The node to list anti affinity rules for
                     One of network, netwok_domain, or node is required
        :type  node: :class:`Node` or ``str``

        :param filter_id: This will allow you to filter the rules
                          by this node id
        :type  filter_id: ``str``

        :type  filter_state: This will allow you to filter rules by
                             node state (i.e. NORMAL)
        :type  filter_state: ``str``

        :rtype: ``list`` of :class:NttCisAntiAffinityRule`
        """
    not_none_arguments = [key for key in (network, network_domain, node) if key is not None]
    if len(not_none_arguments) != 1:
        raise ValueError('One and ONLY one of network, network_domain, or node must be set')
    params = {}
    if network_domain is not None:
        params['networkDomainId'] = self._network_domain_to_network_domain_id(network_domain)
    if network is not None:
        params['networkId'] = self._network_to_network_id(network)
    if node is not None:
        params['serverId'] = self._node_to_node_id(node)
    if filter_id is not None:
        params['id'] = filter_id
    if filter_state is not None:
        params['state'] = filter_state
    paged_result = self.connection.paginated_request_with_orgId_api_2('server/antiAffinityRule', method='GET', params=params)
    rules = []
    for result in paged_result:
        rules.extend(self._to_anti_affinity_rules(result))
    return rules