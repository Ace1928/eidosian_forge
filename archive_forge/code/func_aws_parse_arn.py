import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def aws_parse_arn(self, value):
    """Parse and validate string for ARN components.

        :type value: str
        :rtype: dict
        """
    if value is None or not value.startswith('arn:'):
        return None
    try:
        arn_dict = ARN_PARSER.parse_arn(value)
    except InvalidArnException:
        return None
    if not all((arn_dict['partition'], arn_dict['service'], arn_dict['resource'])):
        return None
    arn_dict['accountId'] = arn_dict.pop('account')
    resource = arn_dict.pop('resource')
    arn_dict['resourceId'] = resource.replace(':', '/').split('/')
    return arn_dict