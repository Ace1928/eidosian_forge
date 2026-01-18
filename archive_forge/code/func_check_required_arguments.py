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
def check_required_arguments(argument_spec, parameters, options_context=None):
    """Check all parameters in argument_spec and return a list of parameters
    that are required but not present in parameters.

    Raises :class:`TypeError` if the check fails

    :arg argument_spec: Argument spec dictionary containing all parameters
        and their specification
    :arg parameters: Dictionary of parameters
    :kwarg options_context: List of strings of parent key names if ``argument_spec`` are
        in a sub spec.

    :returns: Empty list or raises :class:`TypeError` if the check fails.
    """
    missing = []
    if argument_spec is None:
        return missing
    for k, v in argument_spec.items():
        required = v.get('required', False)
        if required and k not in parameters:
            missing.append(k)
    if missing:
        msg = 'missing required arguments: %s' % ', '.join(sorted(missing))
        if options_context:
            msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
        raise TypeError(to_native(msg))
    return missing