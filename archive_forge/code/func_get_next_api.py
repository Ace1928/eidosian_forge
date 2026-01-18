from __future__ import (absolute_import, division, print_function)
import logging
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def get_next_api(message):
    """make sure _links is present, and href is present if next is present
               return api if next is present, None otherwise
               return error if _links or href are missing
            """
    api, error = (None, None)
    if message is None or '_links' not in message:
        error = 'Expecting _links key in %s' % message
    elif 'next' in message['_links']:
        if 'href' in message['_links']['next']:
            api = message['_links']['next']['href']
        else:
            error = 'Expecting href key in %s' % message['_links']['next']
    return (api, error)