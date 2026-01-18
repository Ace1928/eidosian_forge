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
def check_type_int(value):
    """Verify that the value is an integer and return it or convert the value
    to an integer and return it

    Raises :class:`TypeError` if unable to convert to an int

    :arg value: String or int to convert of verify

    :return: int of given value
    """
    if isinstance(value, integer_types):
        return value
    if isinstance(value, string_types):
        try:
            return int(value)
        except ValueError:
            pass
    raise TypeError('%s cannot be converted to an int' % type(value))