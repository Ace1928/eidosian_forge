from __future__ import absolute_import, division, print_function
from os import environ
from urllib.parse import urljoin
import platform
def fusion_argument_spec():
    """Return standard base dictionary used for the argument_spec argument in AnsibleModule"""
    return {PARAM_ISSUER_ID: {'no_log': True, 'aliases': [PARAM_APP_ID], 'deprecated_aliases': [{'name': PARAM_APP_ID, 'version': DEP_VER, 'collection_name': 'purefusion.fusion'}]}, PARAM_PRIVATE_KEY_FILE: {'no_log': False, 'aliases': [PARAM_KEY_FILE], 'deprecated_aliases': [{'name': PARAM_KEY_FILE, 'version': DEP_VER, 'collection_name': 'purefusion.fusion'}]}, PARAM_PRIVATE_KEY_PASSWORD: {'no_log': True}, PARAM_ACCESS_TOKEN: {'no_log': True}}