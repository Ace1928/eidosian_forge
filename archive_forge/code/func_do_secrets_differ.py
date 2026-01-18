from __future__ import absolute_import, division, print_function
from ansible.module_utils.six.moves.urllib.parse import quote
from . import errors
def do_secrets_differ(current, desired):
    return set(((c['name'], c['secret']) for c in current.get('secrets') or [])) != set(((d['name'], d['secret']) for d in desired.get('secrets') or []))