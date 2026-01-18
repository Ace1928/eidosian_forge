from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import binary_type, PY3
from ansible.module_utils.six.moves.http_client import responses as http_responses
def format_http_status(status_code):
    expl = http_responses.get(status_code)
    if not expl:
        return str(status_code)
    return '%d %s' % (status_code, expl)