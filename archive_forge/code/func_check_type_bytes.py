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
def check_type_bytes(value):
    """Convert a human-readable string value to bytes

    Raises :class:`TypeError` if unable to covert the value
    """
    try:
        return human_to_bytes(value)
    except ValueError:
        raise TypeError('%s cannot be converted to a Byte value' % type(value))