from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
import os
import ast
import datetime
import shutil
import tempfile
from ansible.module_utils.basic import json
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import filterfalse
from ansible.module_utils.six.moves.urllib.parse import urlencode, urljoin
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.connection import Connection
from ansible_collections.cisco.mso.plugins.module_utils.constants import NDO_API_VERSION_PATH_FORMAT
def query_objs(self, path, key=None, api_version='v1', **kwargs):
    """Query the MSO REST API for objects in a path"""
    found = []
    objs = self.request(path, api_version=api_version, method='GET')
    if objs == {} or objs == []:
        return found
    if key is None:
        key = path
    if isinstance(objs, dict):
        if key not in objs:
            self.fail_json(msg="Key '{0}' missing from data".format(key), data=objs)
        objs_list = objs.get(key)
    else:
        objs_list = objs
    for obj in objs_list:
        for kw_key, kw_value in kwargs.items():
            if kw_value is None:
                continue
            if isinstance(kw_value, dict):
                obj_value = obj.get(kw_key)
                if obj_value is not None and isinstance(obj_value, dict):
                    breakout = False
                    for kw_key_lvl2, kw_value_lvl2 in kw_value.items():
                        if obj_value.get(kw_key_lvl2) != kw_value_lvl2:
                            breakout = True
                            break
                    if breakout:
                        break
                else:
                    break
            elif obj.get(kw_key) != kw_value:
                break
        else:
            found.append(obj)
    return found