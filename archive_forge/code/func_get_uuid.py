import uuid
import socket
import struct
from libcloud.common.base import ConnectionKey
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
def get_uuid(self, unique_field=None):
    """

        :param  unique_field: Unique field
        :type   unique_field: ``bool``
        :rtype: :class:`UUID`
        """
    return str(uuid.uuid4())