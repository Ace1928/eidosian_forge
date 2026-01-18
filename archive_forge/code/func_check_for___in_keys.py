from __future__ import absolute_import, division, print_function
import codecs
import copy
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_bytes, to_native, to_text
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def check_for___in_keys(self, d_param):
    """Method to warn on underscore in a ZAPI tag"""
    if isinstance(d_param, dict):
        for key, val in d_param.items():
            self.check_for___in_keys(val)
            if '_' in key:
                self.warnings.append("Underscore in ZAPI tag: %s, do you mean '-'?" % key)
    elif isinstance(d_param, list):
        for val in d_param:
            self.check_for___in_keys(val)