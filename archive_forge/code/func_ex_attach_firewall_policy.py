import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def ex_attach_firewall_policy(self, policy, node, nic_mac=None):
    """
        Attach firewall policy to a public NIC interface on the server.

        :param policy: Firewall policy to attach.
        :type policy: :class:`.CloudSigmaFirewallPolicy`

        :param node: Node to attach policy to.
        :type node: :class:`libcloud.compute.base.Node`

        :param nic_mac: Optional MAC address of the NIC to add the policy to.
                        If not specified, first public interface is used
                        instead.
        :type nic_mac: ``str``

        :return: Node object to which the policy was attached to.
        :rtype: :class:`libcloud.compute.base.Node`
        """
    nics = copy.deepcopy(node.extra.get('nics', []))
    if nic_mac:
        nic = [n for n in nics if n['mac'] == nic_mac]
    else:
        nic = nics
    if len(nic) == 0:
        raise ValueError('Cannot find the NIC interface to attach a policy to')
    nic = nic[0]
    nic['firewall_policy'] = policy.id
    params = {'nics': nics}
    node = self.ex_edit_node(node=node, params=params)
    return node