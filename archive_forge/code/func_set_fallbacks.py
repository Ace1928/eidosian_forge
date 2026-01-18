from __future__ import absolute_import, division, print_function
import datetime
import os
from collections import deque
from itertools import chain
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.common.warnings import warn
from ansible.module_utils.errors import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six.moves.collections_abc import (
from ansible.module_utils.six import (
from ansible.module_utils.common.validation import (
def set_fallbacks(argument_spec, parameters):
    no_log_values = set()
    for param, value in argument_spec.items():
        fallback = value.get('fallback', (None,))
        fallback_strategy = fallback[0]
        fallback_args = []
        fallback_kwargs = {}
        if param not in parameters and fallback_strategy is not None:
            for item in fallback[1:]:
                if isinstance(item, dict):
                    fallback_kwargs = item
                else:
                    fallback_args = item
            try:
                fallback_value = fallback_strategy(*fallback_args, **fallback_kwargs)
            except AnsibleFallbackNotFound:
                continue
            else:
                if value.get('no_log', False) and fallback_value:
                    no_log_values.add(fallback_value)
                parameters[param] = fallback_value
    return no_log_values