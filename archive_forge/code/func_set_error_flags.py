from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def set_error_flags(self):
    error_flags = self.module.params['continue_on_error']
    generic_flags = ('always', 'never')
    if len(error_flags) > 1:
        for key in generic_flags:
            if key in error_flags:
                self.module.fail_json(msg="%s needs to be the only keyword in 'continue_on_error' option." % key)
    specific_flags = ('rpc_error', 'missing_vserver_api_error', 'key_error', 'other_error')
    for key in error_flags:
        if key not in generic_flags and key not in specific_flags:
            self.module.fail_json(msg="%s is not a valid keyword in 'continue_on_error' option." % key)
    self.error_flags = dict()
    for flag in specific_flags:
        self.error_flags[flag] = True
        for key in error_flags:
            if key == 'always' or key == flag:
                self.error_flags[flag] = False