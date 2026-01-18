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
def check_mutually_exclusive(terms, parameters, options_context=None):
    """Check mutually exclusive terms against argument parameters

    Accepts a single list or list of lists that are groups of terms that should be
    mutually exclusive with one another

    :arg terms: List of mutually exclusive parameters
    :arg parameters: Dictionary of parameters
    :kwarg options_context: List of strings of parent key names if ``terms`` are
        in a sub spec.

    :returns: Empty list or raises :class:`TypeError` if the check fails.
    """
    results = []
    if terms is None:
        return results
    for check in terms:
        count = count_terms(check, parameters)
        if count > 1:
            results.append(check)
    if results:
        full_list = ['|'.join(check) for check in results]
        msg = 'parameters are mutually exclusive: %s' % ', '.join(full_list)
        if options_context:
            msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
        raise TypeError(to_native(msg))
    return results