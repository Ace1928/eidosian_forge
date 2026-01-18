from __future__ import absolute_import, division, print_function
import os
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
from ansible.module_utils.urls import fetch_url
def parse_output_for_keys(output, short_format=False):
    found = []
    lines = to_native(output).split('\n')
    for line in lines:
        if (line.startswith('pub') or line.startswith('sub')) and 'expired' not in line:
            try:
                tokens = line.split()
                code = tokens[1]
                len_type, real_code = code.split('/')
            except (IndexError, ValueError):
                try:
                    tokens = line.split(':')
                    real_code = tokens[4]
                except (IndexError, ValueError):
                    continue
            found.append(real_code)
    if found and short_format:
        found = shorten_key_ids(found)
    return found