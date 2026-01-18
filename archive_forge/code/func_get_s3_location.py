import io
import logging
import math
import re
import urllib
import eventlet
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import units
import glance_store
from glance_store import capabilities
from glance_store.common import utils
import glance_store.driver
from glance_store import exceptions
from glance_store.i18n import _
import glance_store.location
def get_s3_location(s3_host):
    """Get S3 region information from ``s3_store_host``.

    :param s3_host: S3 endpoint url
    :returns: string value; region information which user wants to use on
              Amazon S3, and if user wants to use S3 compatible storage,
              returns ''
    """
    locations = {'s3.amazonaws.com': '', 's3-us-east-1.amazonaws.com': 'us-east-1', 's3-us-east-2.amazonaws.com': 'us-east-2', 's3-us-west-1.amazonaws.com': 'us-west-1', 's3-us-west-2.amazonaws.com': 'us-west-2', 's3-ap-east-1.amazonaws.com': 'ap-east-1', 's3-ap-south-1.amazonaws.com': 'ap-south-1', 's3-ap-northeast-1.amazonaws.com': 'ap-northeast-1', 's3-ap-northeast-2.amazonaws.com': 'ap-northeast-2', 's3-ap-northeast-3.amazonaws.com': 'ap-northeast-3', 's3-ap-southeast-1.amazonaws.com': 'ap-southeast-1', 's3-ap-southeast-2.amazonaws.com': 'ap-southeast-2', 's3-ca-central-1.amazonaws.com': 'ca-central-1', 's3-cn-north-1.amazonaws.com.cn': 'cn-north-1', 's3-cn-northwest-1.amazonaws.com.cn': 'cn-northwest-1', 's3-eu-central-1.amazonaws.com': 'eu-central-1', 's3-eu-west-1.amazonaws.com': 'eu-west-1', 's3-eu-west-2.amazonaws.com': 'eu-west-2', 's3-eu-west-3.amazonaws.com': 'eu-west-3', 's3-eu-north-1.amazonaws.com': 'eu-north-1', 's3-sa-east-1.amazonaws.com': 'sa-east-1'}
    key = re.sub('^(https?://)?(?P<host>[^:]+[^/])(:[0-9]+)?/?$', '\\g<host>', s3_host)
    return locations.get(key, '')