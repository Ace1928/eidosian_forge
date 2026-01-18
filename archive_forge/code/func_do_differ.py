from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def do_differ(current, desired, *ignored_keys):
    if 'metadata' in desired:
        if do_differ(current['metadata'], desired['metadata'], 'created_by'):
            return True
        ignored_keys = ignored_keys + ('metadata',)
    for key, value in desired.items():
        if key in ignored_keys:
            continue
        if value != current.get(key):
            return True
    return False