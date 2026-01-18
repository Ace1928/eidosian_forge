import time
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
from libcloud.compute.types import Provider, NodeState
def ex_update_node_access(self, node, ipaddr):
    """
        Update the remote ip accessing the node.

        :param node: the reservation node to update
        :type  node: :class:`Node`

        :param ipaddr: the ipaddr used to access the node
        :type  ipaddr: ``str``

        :return: node with updated information
        :rtype: :class:`Node`
        """
    return self._to_status(node.id, node.image, ipaddr)