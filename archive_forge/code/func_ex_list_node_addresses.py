import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
def ex_list_node_addresses(self, node):
    """List all IPv4 addresses attached to node

        :param node: Node to list IP addresses
        :type node: :class:`Node`

        :return: LinodeIPAddress list
        :rtype: `list` of :class:`LinodeIPAddress`
        """
    if not isinstance(node, Node):
        raise LinodeExceptionV4('Invalid node instance')
    response = self.connection.request('/v4/linode/instances/%s/ips' % node.id).object
    return self._to_addresses(response)