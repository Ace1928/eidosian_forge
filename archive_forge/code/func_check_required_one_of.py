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
def check_required_one_of(terms, parameters, options_context=None):
    """Check each list of terms to ensure at least one exists in the given module
    parameters

    Accepts a list of lists or tuples

    :arg terms: List of lists of terms to check. For each list of terms, at
        least one is required.
    :arg parameters: Dictionary of parameters
    :kwarg options_context: List of strings of parent key names if ``terms`` are
        in a sub spec.

    :returns: Empty list or raises :class:`TypeError` if the check fails.
    """
    results = []
    if terms is None:
        return results
    for term in terms:
        count = count_terms(term, parameters)
        if count == 0:
            results.append(term)
    if results:
        for term in results:
            msg = 'one of the following is required: %s' % ', '.join(term)
            if options_context:
                msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
            raise TypeError(to_native(msg))
    return results