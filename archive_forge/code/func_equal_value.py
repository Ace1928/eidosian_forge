from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
import logging
def equal_value(existing, parameter):
    if isinstance(existing, str):
        return existing == str(parameter)
    elif isinstance(parameter, str):
        return str(existing) == parameter
    else:
        return existing == parameter