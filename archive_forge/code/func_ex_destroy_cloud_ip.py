import base64
from libcloud.utils.py3 import b, httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
from libcloud.common.brightbox import BrightboxConnection
def ex_destroy_cloud_ip(self, cloud_ip_id):
    """
        Release the cloud IP address from the account's ownership

        @note: This is an API extension for use on Brightbox

        :param  cloud_ip_id: The id of the cloud ip.
        :type   cloud_ip_id: ``str``

        :return: True if the unmap was successful.
        :rtype: ``bool``
        """
    response = self.connection.request('/{}/cloud_ips/{}'.format(self.api_version, cloud_ip_id), method='DELETE')
    return response.status == httplib.OK