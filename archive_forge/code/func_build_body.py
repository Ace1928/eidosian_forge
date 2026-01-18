from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
def build_body(self, action):
    keys = {'create': ['access', 'access_control', 'advanced_rights', 'apply_to', 'ignore_paths', 'propagation_mode', 'rights', 'user'], 'modify': ['access', 'access_control', 'advanced_rights', 'apply_to', 'ignore_paths', 'rights'], 'delete': ['access', 'access_control', 'apply_to', 'ignore_paths', 'propagation_mode']}
    if action not in keys:
        self.module.fail_json(msg='Internal error - unexpected action %s' % action)
    body = {}
    for key in keys[action]:
        if key in self.parameters:
            body[key] = self.parameters[key]
    return body