from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def key_manager_setup(self):
    """
        set up external key manager.
        deprecated as of ONTAP 9.6.
        """
    key_manager_setup = netapp_utils.zapi.NaElement('security-key-manager-setup')
    try:
        self.cluster.invoke_successfully(key_manager_setup, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error setting up key manager: %s' % to_native(error), exception=traceback.format_exc())