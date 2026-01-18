import json
import time
import datetime
from libcloud.utils.py3 import basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
def ex_node_operation(self, node, operation, wait=True):
    """
        Run custom operations on the node

        :param node:     the node to run operation on
        :type node: :class:`Node`

        :param operation:   the operation to run
        :type operation:   ``str``

        :param ex_wait:     wait for destroy to complete (optional)
        :type ex_wait:      ``bool``

        :rtype: ``bool``
        """
    if node.id:
        request_data = {'id': node.id}
    elif node.name:
        request_data = {'name': node.name}
    else:
        raise ValueError('Invalid node for %s node operation: missing id / name' % operation)
    if operation == 'terminate':
        request_data['force'] = True
    command_id = self.connection.request('/service/server/%s' % operation, method='POST', data=json.dumps(request_data)).object[0]
    if wait:
        self.ex_wait_command(command_id)
    else:
        node.extra['%s_command_id' % operation] = command_id
    return True