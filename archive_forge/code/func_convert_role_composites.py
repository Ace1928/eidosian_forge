from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def convert_role_composites(self, composites):
    keycloak_compatible_composites = {'client': {}, 'realm': []}
    for composite in composites:
        if 'state' not in composite or composite['state'] == 'present':
            if 'client_id' in composite and composite['client_id'] is not None:
                if composite['client_id'] not in keycloak_compatible_composites['client']:
                    keycloak_compatible_composites['client'][composite['client_id']] = []
                keycloak_compatible_composites['client'][composite['client_id']].append(composite['name'])
            else:
                keycloak_compatible_composites['realm'].append(composite['name'])
    return keycloak_compatible_composites