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
def ex_describe_quotas(self, dry_run=False, filters=None, max_results=None, marker=None):
    """
        Describes one or more of your quotas.

        :param      dry_run: dry_run
        :type       dry_run: ``bool``

        :param      filters: The filters so that the response returned includes
                             information for certain quotas only.
        :type       filters: ``dict``

        :param      max_results: The maximum number of items that can be
                                 returned in a single page (by default, 100)
        :type       max_results: ``int``

        :param      marker: Set quota marker
        :type       marker: ``string``

        :return:    (is_truncated, quota) tuple
        :rtype:     ``(bool, dict)``
        """
    if filters:
        raise NotImplementedError('quota filters are not implemented')
    if marker:
        raise NotImplementedError('quota marker is not implemented')
    params = {'Action': 'DescribeQuotas'}
    if dry_run:
        params.update({'DryRun': dry_run})
    if max_results:
        params.update({'MaxResults': max_results})
    response = self.connection.request(self.path, params=params, method='GET').object
    quota = self._to_quota(response)
    is_truncated = findtext(element=response, xpath='isTruncated', namespace=OUTSCALE_NAMESPACE)
    return (is_truncated, quota)