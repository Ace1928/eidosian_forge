from __future__ import absolute_import, division, print_function
import base64
import traceback
from ansible.module_utils.basic import AnsibleModule, sanitize_keys
from ansible_collections.community.general.plugins.module_utils.ipa import IPAClient, ipa_argument_spec
from ansible.module_utils.common.text.converters import to_native
def base32_to_base64(base32_string):
    """Converts base32 string to base64 string"""
    b64_string = base64.b64encode(base64.b32decode(base32_string)).decode('ascii')
    return b64_string