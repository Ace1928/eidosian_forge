import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_attach_nic_to_node(self, node, network, ip_address=None):
    """
        Add an extra Nic to a VM

        :param  network: NetworkOffering object
        :type   network: :class:'CloudStackNetwork`

        :param  node: Node Object
        :type   node: :class:'CloudStackNode`

        :param  ip_address: Optional, specific IP for this Nic
        :type   ip_address: ``str``


        :rtype: ``bool``
        """
    args = {'virtualmachineid': node.id, 'networkid': network.id}
    if ip_address is not None:
        args['ipaddress'] = ip_address
    self._async_request(command='addNicToVirtualMachine', params=args)
    return True