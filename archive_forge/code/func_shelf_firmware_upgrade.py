from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def shelf_firmware_upgrade(self):
    """
        Upgrade shelf firmware image
        """
    shelf_firmware_update_info = netapp_utils.zapi.NaElement('storage-shelf-firmware-update')
    try:
        self.server.invoke_successfully(shelf_firmware_update_info, enable_tunneling=True)
        return True
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error updating shelf firmware image : %s' % to_native(error), exception=traceback.format_exc())