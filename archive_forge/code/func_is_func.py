import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def is_func(self, argument):
    """Determine if an object is a function object.

        :type argument: Any
        :rtype: bool
        """
    return isinstance(argument, dict) and 'fn' in argument