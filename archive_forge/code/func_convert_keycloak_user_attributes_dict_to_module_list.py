from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def convert_keycloak_user_attributes_dict_to_module_list(self, attributes):
    module_attributes_list = []
    for key in attributes:
        attr = {}
        attr['name'] = key
        attr['values'] = attributes[key]
        module_attributes_list.append(attr)
    return module_attributes_list