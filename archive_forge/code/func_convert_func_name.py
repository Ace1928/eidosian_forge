import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def convert_func_name(self, value):
    """Normalize function names.

        :type value: str
        :rtype: str
        """
    normalized_name = f'{xform_name(value)}'
    if normalized_name == 'not':
        normalized_name = f'_{normalized_name}'
    return normalized_name.replace('.', '_')