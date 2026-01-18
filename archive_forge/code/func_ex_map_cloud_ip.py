import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_map_cloud_ip(self, cloud_ip_id, interface_id):
    """
        Maps (or points) a cloud IP address at a server's interface
        or a load balancer to allow them to respond to public requests

        @note: This is an API extension for use on Brightbox

        :param  cloud_ip_id: The id of the cloud ip.
        :type   cloud_ip_id: ``str``

        :param  interface_id: The Interface ID or LoadBalancer ID to
                              which this Cloud IP should be mapped to
        :type   interface_id: ``str``

        :return: True if the mapping was successful.
        :rtype: ``bool``
        """
    response = self._post('/{}/cloud_ips/{}/map'.format(self.api_version, cloud_ip_id), {'destination': interface_id})
    return response.status == httplib.ACCEPTED