import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_create_cloud_ip(self, reverse_dns=None):
    """
        Requests a new cloud IP address for the account

        @note: This is an API extension for use on Brightbox

        :param      reverse_dns: Reverse DNS hostname
        :type       reverse_dns: ``str``

        :rtype: ``dict``
        """
    params = {}
    if reverse_dns:
        params['reverse_dns'] = reverse_dns
    return self._post('/%s/cloud_ips' % self.api_version, params).object