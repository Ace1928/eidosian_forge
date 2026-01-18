from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def shelf_firmware_info_get(self):
    """
        Get the current firmware of shelf module
        :return:dict with module id and firmware info
        """
    shelf_id_fw_info = {}
    shelf_firmware_info_get = netapp_utils.zapi.NaElement('storage-shelf-info-get-iter')
    desired_attributes = netapp_utils.zapi.NaElement('desired-attributes')
    storage_shelf_info = netapp_utils.zapi.NaElement('storage-shelf-info')
    shelf_module = netapp_utils.zapi.NaElement('shelf-modules')
    shelf_module_info = netapp_utils.zapi.NaElement('storage-shelf-module-info')
    shelf_module.add_child_elem(shelf_module_info)
    storage_shelf_info.add_child_elem(shelf_module)
    desired_attributes.add_child_elem(storage_shelf_info)
    shelf_firmware_info_get.add_child_elem(desired_attributes)
    try:
        result = self.server.invoke_successfully(shelf_firmware_info_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching shelf module firmware  details: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        shelf_info = result.get_child_by_name('attributes-list').get_child_by_name('storage-shelf-info')
        if shelf_info.get_child_by_name('shelf-modules') and shelf_info.get_child_by_name('shelf-modules').get_child_by_name('storage-shelf-module-info'):
            shelves = shelf_info['shelf-modules'].get_children()
            for shelf in shelves:
                shelf_id_fw_info[shelf.get_child_content('module-id')] = shelf.get_child_content('module-fw-revision')
    return shelf_id_fw_info