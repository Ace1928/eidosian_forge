import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_apply_tag_to_asset(self, asset, tag_key, value=None):
    """
        Apply a tag to a NTTC-CIS Asset

        :param asset: The asset to apply a tag to. (required)
        :type  asset: :class:`Node` or :class:`NodeImage` or
                      :class:`NttCisNewtorkDomain` or
                      :class:`NttCisVlan` or
                      :class:`NttCisPublicIpBlock`

        :param tag_key: The tag_key to apply to the asset. (required)
        :type  tag_key: :class:`NttCisTagKey` or ``str``

        :param value: The value to be assigned to the tag key
                      This is only required if the :class:`NttCisTagKey`
                      requires it
        :type  value: ``str``

        :rtype: ``bool``
        """
    asset_type = self._get_tagging_asset_type(asset)
    tag_key_name = self._tag_key_to_tag_key_name(tag_key)
    apply_tags = ET.Element('applyTags', {'xmlns': TYPES_URN})
    ET.SubElement(apply_tags, 'assetType').text = asset_type
    ET.SubElement(apply_tags, 'assetId').text = asset.id
    tag_ele = ET.SubElement(apply_tags, 'tag')
    ET.SubElement(tag_ele, 'tagKeyName').text = tag_key_name
    if value is not None:
        ET.SubElement(tag_ele, 'value').text = value
    response = self.connection.request_with_orgId_api_2('tag/applyTags', method='POST', data=ET.tostring(apply_tags)).object
    response_code = findtext(response, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']