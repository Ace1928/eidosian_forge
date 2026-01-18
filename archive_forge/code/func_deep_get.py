from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.http_client import HTTPException
import json
import logging
def deep_get(dct, dotted_path, default=_empty, use_reference_table=True):
    result_dct = {}
    for key in dotted_path.split('.'):
        try:
            key_field = key
            if use_reference_table:
                key_field = referenced_value(key, cyberark_reference_fieldnames, default=key)
            if len(list(result_dct.keys())) == 0:
                result_dct = dct
            logging.debug('keys=%s key_field=>%s   key=>%s', ','.join(list(result_dct.keys())), key_field, key)
            result_dct = result_dct[key_field] if key_field in list(result_dct.keys()) else result_dct[key]
            if result_dct is None:
                return default
        except KeyError as e:
            logging.debug('KeyError %s', to_text(e))
            if default is _empty:
                raise
            return default
    return result_dct