from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
def sanitize_arguments(arguments):
    for key, value in list(arguments.items()):
        if value is None:
            del arguments[key]
    return arguments