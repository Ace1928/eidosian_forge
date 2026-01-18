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
def ex_describe_volumes_modifications(self, dry_run=False, volume_ids=None, filters=None):
    """
        Describes one or more of your volume modifications.

        :param      dry_run: dry_run
        :type       dry_run: ``bool``

        :param      volume_ids: The volume_ids so that the response includes
                             information for only said volumes
        :type       volume_ids: ``dict``

        :param      filters: The filters so that the response includes
                             information for only certain volumes
        :type       filters: ``dict``

        :return:  List of volume modification status objects
        :rtype:   ``list`` of :class:`VolumeModification
        """
    params = {'Action': 'DescribeVolumesModifications'}
    if dry_run:
        params.update({'DryRun': dry_run})
    if volume_ids:
        params.update(self._pathlist('VolumeId', volume_ids))
    if filters:
        params.update(self._build_filters(filters))
    response = self.connection.request(self.path, params=params).object
    return self._to_volume_modifications(response)