import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
def ex_detach_nic_from_node(self, nic, node):
    """
        Remove Nic from a VM

        :param  nic: Nic object
        :type   nic: :class:'CloudStackNetwork`

        :param  node: Node Object
        :type   node: :class:'CloudStackNode`

        :rtype: ``bool``
        """
    self._async_request(command='removeNicFromVirtualMachine', params={'nicid': nic.id, 'virtualmachineid': node.id})
    return True