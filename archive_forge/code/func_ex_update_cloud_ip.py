import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_update_cloud_ip(self, cloud_ip_id, reverse_dns):
    """
        Update some details of the cloud IP address

        @note: This is an API extension for use on Brightbox

        :param  cloud_ip_id: The id of the cloud ip.
        :type   cloud_ip_id: ``str``

        :param      reverse_dns: Reverse DNS hostname
        :type       reverse_dns: ``str``

        :rtype: ``dict``
        """
    response = self._put('/{}/cloud_ips/{}'.format(self.api_version, cloud_ip_id), {'reverse_dns': reverse_dns})
    return response.status == httplib.OK