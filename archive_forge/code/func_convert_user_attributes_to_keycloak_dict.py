from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def convert_user_attributes_to_keycloak_dict(self, attributes):
    keycloak_user_attributes_dict = {}
    for attribute in attributes:
        if ('state' not in attribute or attribute['state'] == 'present') and 'name' in attribute:
            keycloak_user_attributes_dict[attribute['name']] = attribute['values'] if 'values' in attribute else []
    return keycloak_user_attributes_dict