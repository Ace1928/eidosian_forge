import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def resolve_headers(self, scope_vars, rule_lib):
    """Iterate through headers attribute resolving all values.

        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: dict
        """
    resolved_headers = {}
    headers = self.endpoint.get('headers', {})
    for header, values in headers.items():
        resolved_headers[header] = [rule_lib.resolve_value(item, scope_vars) for item in values]
    return resolved_headers