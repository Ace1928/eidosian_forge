import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_remove_tag_from_asset(self, asset, tag_key):
    """
        Remove a tag from an asset

        :param asset: The asset to remove a tag from. (required)
        :type  asset: :class:`Node` or :class:`NodeImage` or
                      :class:`NttCisNewtorkDomain` or
                      :class:`NttCisVlan` or
                      :class:`NttCisPublicIpBlock`

        :param tag_key: The tag key you want to remove (required)
        :type  tag_key: :class:`NttCisTagKey` or ``str``

        :rtype: ``bool``
        """
    asset_type = self._get_tagging_asset_type(asset)
    tag_key_name = self._tag_key_to_tag_key_name(tag_key)
    apply_tags = ET.Element('removeTags', {'xmlns': TYPES_URN})
    ET.SubElement(apply_tags, 'assetType').text = asset_type
    ET.SubElement(apply_tags, 'assetId').text = asset.id
    ET.SubElement(apply_tags, 'tagKeyName').text = tag_key_name
    response = self.connection.request_with_orgId_api_2('tag/removeTags', method='POST', data=ET.tostring(apply_tags)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']