import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def aws_is_virtual_hostable_s3_bucket(self, value, allow_subdomains):
    """Evaluates whether a value is a valid bucket name for virtual host
        style bucket URLs. To pass, the value must meet the following criteria:
        1. is_valid_host_label(value) is True
        2. length between 3 and 63 characters (inclusive)
        3. does not contain uppercase characters
        4. is not formatted as an IP address

        If allow_subdomains is True, split on `.` and validate
        each component separately.

        :type value: str
        :type allow_subdomains: bool
        :rtype: bool
        """
    if value is None or len(value) < 3 or value.lower() != value or (IPV4_RE.match(value) is not None):
        return False
    return self.is_valid_host_label(value, allow_subdomains=allow_subdomains)