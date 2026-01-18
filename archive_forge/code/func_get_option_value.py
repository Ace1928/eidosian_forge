from __future__ import (absolute_import, division, print_function)
import base64
from ansible.errors import AnsibleLookupError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.sops.plugins.module_utils.sops import Sops, SopsError
from ansible.utils.display import Display
def get_option_value(argument_name):
    return self.get_option(argument_name)