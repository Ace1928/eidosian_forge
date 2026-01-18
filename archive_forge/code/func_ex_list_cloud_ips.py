import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_list_cloud_ips(self):
    """
        List Cloud IPs

        @note: This is an API extension for use on Brightbox

        :rtype: ``list`` of ``dict``
        """
    return self.connection.request('/%s/cloud_ips' % self.api_version).object