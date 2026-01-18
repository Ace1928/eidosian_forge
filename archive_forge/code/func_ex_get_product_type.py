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
def ex_get_product_type(self, image_id, snapshot_id=None):
    """
        Gets the product type of a specified OMI or snapshot.

        :param      image_id: The ID of the OMI
        :type       image_id: ``string``

        :param      snapshot_id: The ID of the snapshot
        :type       snapshot_id: ``string``

        :return:    A product type
        :rtype:     ``dict``
        """
    params = {'Action': 'GetProductType'}
    params.update({'ImageId': image_id})
    if snapshot_id is not None:
        params.update({'SnapshotId': snapshot_id})
    response = self.connection.request(self.path, params=params, method='GET').object
    product_type = self._to_product_type(response)
    return product_type