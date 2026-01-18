from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def prepare_object(self, existing, obj):
    operational_attributes = {'CreateIndex', 'CreateTime', 'Hash', 'ModifyIndex'}
    existing = {k: v for k, v in existing.items() if k not in operational_attributes}
    for k, v in obj.items():
        existing[k] = v
    return existing