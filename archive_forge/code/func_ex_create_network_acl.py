import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_network_acl(self, protocol, acl_id, cidr_list, start_port, end_port, action=None, traffic_type=None):
    """
        Creates an ACL rule in the given network (the network has to belong to
        VPC)

        :param      protocol: the protocol for the ACL rule. Valid values are
                    TCP/UDP/ICMP/ALL or valid protocol number
        :type       protocol: ``string``

        :param      acl_id: Name of the network ACL List
        :type       acl_id: ``str``

        :param      cidr_list: the cidr list to allow traffic from/to
        :type       cidr_list: ``str``

        :param      start_port: the starting port of ACL
        :type       start_port: ``str``

        :param      end_port: the ending port of ACL
        :type       end_port: ``str``

        :param      action: scl entry action, allow or deny
        :type       action: ``str``

        :param      traffic_type: the traffic type for the ACL,can be Ingress
                    or Egress, defaulted to Ingress if not specified
        :type       traffic_type: ``str``

        :rtype: :class:`CloudStackNetworkACL`
        """
    args = {'protocol': protocol, 'aclid': acl_id, 'cidrlist': cidr_list, 'startport': start_port, 'endport': end_port}
    if action:
        args['action'] = action
    else:
        action = 'allow'
    if traffic_type:
        args['traffictype'] = traffic_type
    result = self._async_request(command='createNetworkACL', params=args, method='GET')
    acl = CloudStackNetworkACL(result['networkacl']['id'], protocol, acl_id, action, cidr_list, start_port, end_port, traffic_type)
    return acl