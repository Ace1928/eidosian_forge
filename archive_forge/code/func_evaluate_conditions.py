import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def evaluate_conditions(self, scope_vars, rule_lib):
    """Determine if all conditions in a rule are met.

        :type scope_vars: dict
        :type rule_lib: RuleSetStandardLibrary
        :rtype: bool
        """
    for func_signature in self.conditions:
        result = rule_lib.call_function(func_signature, scope_vars)
        if result is False or result is None:
            return False
    return True