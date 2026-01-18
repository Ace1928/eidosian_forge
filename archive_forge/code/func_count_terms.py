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
def count_terms(terms, parameters):
    """Count the number of occurrences of a key in a given dictionary

    :arg terms: String or iterable of values to check
    :arg parameters: Dictionary of parameters

    :returns: An integer that is the number of occurrences of the terms values
        in the provided dictionary.
    """
    if not is_iterable(terms):
        terms = [terms]
    return len(set(terms).intersection(parameters))