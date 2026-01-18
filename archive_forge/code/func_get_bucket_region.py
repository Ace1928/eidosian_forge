import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def get_bucket_region(self, bucket, response):
    """
        There are multiple potential sources for the new region to redirect to,
        but they aren't all universally available for use. This will try to
        find region from response elements, but will fall back to calling
        HEAD on the bucket if all else fails.

        :param bucket: The bucket to find the region for. This is necessary if
            the region is not available in the error response.
        :param response: A response representing a service request that failed
            due to incorrect region configuration.
        """
    service_response = response[1]
    response_headers = service_response['ResponseMetadata']['HTTPHeaders']
    if 'x-amz-bucket-region' in response_headers:
        return response_headers['x-amz-bucket-region']
    region = service_response.get('Error', {}).get('Region', None)
    if region is not None:
        return region
    try:
        response = self._client.head_bucket(Bucket=bucket)
        headers = response['ResponseMetadata']['HTTPHeaders']
    except ClientError as e:
        headers = e.response['ResponseMetadata']['HTTPHeaders']
    region = headers.get('x-amz-bucket-region', None)
    return region