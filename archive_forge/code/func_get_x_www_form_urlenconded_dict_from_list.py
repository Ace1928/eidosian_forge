from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import fetch_url, open_url
import json
import time
def get_x_www_form_urlenconded_dict_from_list(key, values):
    """Return a dictionary with keys values"""
    if len(values) == 1:
        return {'{key}[]'.format(key=key): values[0]}
    else:
        return dict((('{key}[{index}]'.format(key=key, index=i), x) for i, x in enumerate(values)))