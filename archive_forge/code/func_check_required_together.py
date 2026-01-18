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
def check_required_together(terms, parameters, options_context=None):
    """Check each list of terms to ensure every parameter in each list exists
    in the given parameters.

    Accepts a list of lists or tuples.

    :arg terms: List of lists of terms to check. Each list should include
        parameters that are all required when at least one is specified
        in the parameters.
    :arg parameters: Dictionary of parameters
    :kwarg options_context: List of strings of parent key names if ``terms`` are
        in a sub spec.

    :returns: Empty list or raises :class:`TypeError` if the check fails.
    """
    results = []
    if terms is None:
        return results
    for term in terms:
        counts = [count_terms(field, parameters) for field in term]
        non_zero = [c for c in counts if c > 0]
        if len(non_zero) > 0:
            if 0 in counts:
                results.append(term)
    if results:
        for term in results:
            msg = 'parameters are required together: %s' % ', '.join(term)
            if options_context:
                msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
            raise TypeError(to_native(msg))
    return results