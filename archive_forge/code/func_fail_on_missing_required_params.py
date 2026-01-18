from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def fail_on_missing_required_params(self, action):
    missing_keys = [key for key in ('client_match', 'ro_rule', 'rw_rule') if self.parameters.get(key) is None]
    plural = 's' if len(missing_keys) > 1 else ''
    if missing_keys:
        self.module.fail_json(msg='Error: Missing required option%s for %s export policy rule: %s' % (plural, action, ', '.join(missing_keys)))