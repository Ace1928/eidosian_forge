import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_create_port_forwarding_rule(self, node, address, private_port, public_port, protocol, public_end_port=None, private_end_port=None, openfirewall=True, network_id=None):
    """
        Creates a Port Forwarding Rule, used for Source NAT

        :param  address: IP address of the Source NAT
        :type   address: :class:`CloudStackAddress`

        :param  private_port: Port of the virtual machine
        :type   private_port: ``int``

        :param  protocol: Protocol of the rule
        :type   protocol: ``str``

        :param  public_port: Public port on the Source NAT address
        :type   public_port: ``int``

        :param  node: The virtual machine
        :type   node: :class:`CloudStackNode`

        :param  network_id: The network of the vm the Port Forwarding rule
                            will be created for. Required when public Ip
                            address is not associated with any Guest
                            network yet (VPC case)
        :type   network_id: ``string``

        :rtype: :class:`CloudStackPortForwardingRule`
        """
    args = {'ipaddressid': address.id, 'protocol': protocol, 'privateport': int(private_port), 'publicport': int(public_port), 'virtualmachineid': node.id, 'openfirewall': openfirewall}
    if public_end_port:
        args['publicendport'] = int(public_end_port)
    if private_end_port:
        args['privateendport'] = int(private_end_port)
    if network_id:
        args['networkid'] = network_id
    result = self._async_request(command='createPortForwardingRule', params=args, method='GET')
    rule = CloudStackPortForwardingRule(node, result['portforwardingrule']['id'], address, protocol, public_port, private_port, public_end_port, private_end_port, network_id)
    node.extra['port_forwarding_rules'].append(rule)
    node.public_ips.append(address.address)
    return rule