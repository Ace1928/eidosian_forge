from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def get_rest_items(rest_obj, uri='DeviceService/Devices', key='Id', value='Identifier', selector='value'):
    item_dict = {}
    resp = rest_obj.get_all_items_with_pagination(uri)
    if resp.get(selector):
        item_dict = dict(((item.get(key), item.get(value)) for item in resp[selector]))
    return item_dict