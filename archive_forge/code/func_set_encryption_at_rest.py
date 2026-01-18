from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def set_encryption_at_rest(self, state=None):
    """
        enable/disable encryption at rest
        """
    try:
        if state == 'present':
            encryption_state = 'enable'
            self.sfe.enable_encryption_at_rest()
        elif state == 'absent':
            encryption_state = 'disable'
            self.sfe.disable_encryption_at_rest()
    except Exception as exception_object:
        self.module.fail_json(msg='Failed to %s rest encryption %s' % (encryption_state, to_native(exception_object)), exception=traceback.format_exc())