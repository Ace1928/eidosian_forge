from __future__ import absolute_import, division, print_function
import os
import re
from ast import literal_eval
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common._json_compat import json
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import (
def check_type_float(value):
    """Verify that value is a float or convert it to a float and return it

    Raises :class:`TypeError` if unable to convert to a float

    :arg value: float, int, str, or bytes to verify or convert and return.

    :returns: float of given value.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, (binary_type, text_type, int)):
        try:
            return float(value)
        except ValueError:
            pass
    raise TypeError('%s cannot be converted to a float' % type(value))