import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_destroy_ip(self, ip):
    """
        Delete an ip.

        :param ip: IP object.
        :type ip: :class:`.GridscaleIp`

        :return: ``True`` if delete_image was successful, ``False`` otherwise.
        :rtype: ``bool``
        """
    result = self._sync_request(endpoint='objects/ips/{}'.format(ip.id), method='DELETE')
    return result.status == 204