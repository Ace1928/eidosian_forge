import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def is_valid_host_label(self, value, allow_subdomains):
    """Evaluates whether a value is a valid host label per
        RFC 1123. If allow_subdomains is True, split on `.` and validate
        each component separately.

        :type value: str
        :type allow_subdomains: bool
        :rtype: bool
        """
    if value is None or (allow_subdomains is False and value.count('.') > 0):
        return False
    if allow_subdomains is True:
        return all((self.is_valid_host_label(label, False) for label in value.split('.')))
    return VALID_HOST_LABEL_RE.match(value) is not None