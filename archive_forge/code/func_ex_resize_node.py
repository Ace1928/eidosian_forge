import json
import warnings
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.digitalocean import DigitalOcean_v1_Error, DigitalOcean_v2_BaseDriver
def ex_resize_node(self, node, size):
    """
        Resize the node to a different machine size.  Note that some resize
        operations are reversible, and others are irreversible.

        :param node: Node to rebuild
        :type node: :class:`Node`

        :param size: New size for this machine
        :type node: :class:`NodeSize`

        :return True if the operation began successfully
        :rtype ``bool``
        """
    attr = {'type': 'resize', 'size': size.name}
    res = self.connection.request('/v2/droplets/%s/actions' % node.id, data=json.dumps(attr), method='POST')
    return res.status == httplib.CREATED