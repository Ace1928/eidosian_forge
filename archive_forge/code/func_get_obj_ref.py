from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def get_obj_ref(self, obj):
    """returns reference url from dict object"""
    if not obj:
        return None
    if isinstance(obj, Response):
        obj = json.loads(obj.text)
    if obj.get(0, None):
        return obj[0]['url']
    elif obj.get('url', None):
        return obj['url']
    elif obj.get('results', None):
        return obj['results'][0]['url']
    else:
        return None