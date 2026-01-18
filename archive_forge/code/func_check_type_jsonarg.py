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
def check_type_jsonarg(value):
    """Return a jsonified string. Sometimes the controller turns a json string
    into a dict/list so transform it back into json here

    Raises :class:`TypeError` if unable to covert the value

    """
    if isinstance(value, (text_type, binary_type)):
        return value.strip()
    elif isinstance(value, (list, tuple, dict)):
        return jsonify(value)
    raise TypeError('%s cannot be converted to a json string' % type(value))