import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
def get_node_state(self, node_id):
    """
        Get the state of the node.

        :param  node_id: Id of the Node
        :type   node_id: ``int``

        :rtype: ``str``
        """
    action = '1.2/server/{}'.format(node_id)
    try:
        response = self.connection.request(action)
        return response.object['server']['state']
    except BaseHTTPError as e:
        if e.code == 404:
            return None
        raise