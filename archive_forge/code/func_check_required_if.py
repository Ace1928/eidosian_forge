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
def check_required_if(requirements, parameters, options_context=None):
    """Check parameters that are conditionally required

    Raises :class:`TypeError` if the check fails

    :arg requirements: List of lists specifying a parameter, value, parameters
        required when the given parameter is the specified value, and optionally
        a boolean indicating any or all parameters are required.

    :Example:

    .. code-block:: python

        required_if=[
            ['state', 'present', ('path',), True],
            ['someint', 99, ('bool_param', 'string_param')],
        ]

    :arg parameters: Dictionary of parameters

    :returns: Empty list or raises :class:`TypeError` if the check fails.
        The results attribute of the exception contains a list of dictionaries.
        Each dictionary is the result of evaluating each item in requirements.
        Each return dictionary contains the following keys:

            :key missing: List of parameters that are required but missing
            :key requires: 'any' or 'all'
            :key parameter: Parameter name that has the requirement
            :key value: Original value of the parameter
            :key requirements: Original required parameters

        :Example:

        .. code-block:: python

            [
                {
                    'parameter': 'someint',
                    'value': 99
                    'requirements': ('bool_param', 'string_param'),
                    'missing': ['string_param'],
                    'requires': 'all',
                }
            ]

    :kwarg options_context: List of strings of parent key names if ``requirements`` are
        in a sub spec.
    """
    results = []
    if requirements is None:
        return results
    for req in requirements:
        missing = {}
        missing['missing'] = []
        max_missing_count = 0
        is_one_of = False
        if len(req) == 4:
            key, val, requirements, is_one_of = req
        else:
            key, val, requirements = req
        if is_one_of:
            max_missing_count = len(requirements)
            missing['requires'] = 'any'
        else:
            missing['requires'] = 'all'
        if key in parameters and parameters[key] == val:
            for check in requirements:
                count = count_terms(check, parameters)
                if count == 0:
                    missing['missing'].append(check)
        if len(missing['missing']) and len(missing['missing']) >= max_missing_count:
            missing['parameter'] = key
            missing['value'] = val
            missing['requirements'] = requirements
            results.append(missing)
    if results:
        for missing in results:
            msg = '%s is %s but %s of the following are missing: %s' % (missing['parameter'], missing['value'], missing['requires'], ', '.join(missing['missing']))
            if options_context:
                msg = '{0} found in {1}'.format(msg, ' -> '.join(options_context))
            raise TypeError(to_native(msg))
    return results