from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def get_options_with_value(self, arguments):
    ret_arguments = dict()
    for arg_key, arg_value in arguments.items():
        if arguments[arg_key] is not None:
            ret_arguments[arg_key] = arg_value
    return ret_arguments