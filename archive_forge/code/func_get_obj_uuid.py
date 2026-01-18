from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def get_obj_uuid(self, obj):
    """returns uuid from dict object"""
    if not obj:
        raise ObjectNotFound('Object %s Not found' % obj)
    if isinstance(obj, Response):
        obj = json.loads(obj.text)
    if obj.get(0, None):
        return obj[0]['uuid']
    elif obj.get('uuid', None):
        return obj['uuid']
    elif obj.get('results', None):
        return obj['results'][0]['uuid']
    else:
        return None