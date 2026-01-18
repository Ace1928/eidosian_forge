import logging
import re
from enum import Enum
from string import Formatter
from typing import NamedTuple
from botocore import xform_name
from botocore.compat import IPV4_RE, quote, urlparse
from botocore.exceptions import EndpointResolutionError
from botocore.utils import (
def process_input_parameters(self, input_params):
    """Process each input parameter against its spec.

        :type input_params: dict
        """
    for name, spec in self.parameters.items():
        value = spec.process_input(input_params.get(name))
        if value is not None:
            input_params[name] = value
    return None