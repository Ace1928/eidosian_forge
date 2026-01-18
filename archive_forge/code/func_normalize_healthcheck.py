from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def normalize_healthcheck(healthcheck, normalize_test=False):
    """
    Return dictionary of healthcheck parameters.
    """
    result = dict()
    options = ('test', 'interval', 'timeout', 'start_period', 'retries')
    duration_options = ('interval', 'timeout', 'start_period')
    for key in options:
        if key in healthcheck:
            value = healthcheck[key]
            if value is None:
                continue
            if key in duration_options:
                value = convert_duration_to_nanosecond(value)
            if not value:
                continue
            if key == 'retries':
                try:
                    value = int(value)
                except ValueError:
                    raise ValueError('Cannot parse number of retries for healthcheck. Expected an integer, got "{0}".'.format(value))
            if key == 'test' and normalize_test:
                value = normalize_healthcheck_test(value)
            result[key] = value
    return result