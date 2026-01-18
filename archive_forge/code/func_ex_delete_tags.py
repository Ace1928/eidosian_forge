import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
def ex_delete_tags(self, resource, tags):
    """
        Deletes tags from a resource.

        :param resource: The resource to be tagged
        :type resource: :class:`Node` or :class:`StorageVolume`

        :param tags: A dictionary or other mapping of strings to strings,
                     specifying the tag names and tag values to be deleted.
        :type tags: ``dict``

        :rtype: ``bool``
        """
    if not tags:
        return
    params = {'Action': 'DeleteTags', 'ResourceId.0': resource.id}
    for i, key in enumerate(tags):
        params['Tag.%d.Key' % i] = key
        if tags[key] is not None:
            params['Tag.%d.Value' % i] = tags[key]
    res = self.connection.request(self.path, params=params.copy()).object
    return self._get_boolean(res)