import time
from libcloud.compute.base import (
from libcloud.utils.iso8601 import parse_date
from libcloud.common.gridscale import GridscaleBaseDriver, GridscaleConnection
from libcloud.compute.providers import Provider
def ex_link_isoimage_to_node(self, node, isoimage):
    """
        link and isoimage to a node

        :param node: Node you want to link the iso image to
        :type node: ``object``

        :param isoimage: isomiage you want to link
        :type isoimage: ``object``

        :return: None -> success
        :rtype: ``None``
        """
    result = self._sync_request(data={'object_uuid': isoimage.id}, endpoint='objects/servers/{}/isoimages/'.format(node.id), method='POST')
    return result