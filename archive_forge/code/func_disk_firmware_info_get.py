from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def disk_firmware_info_get(self):
    """
        Get the current firmware of disks module
        :return:
        """
    disk_id_fw_info = {}
    disk_firmware_info_get = netapp_utils.zapi.NaElement('storage-disk-get-iter')
    desired_attributes = netapp_utils.zapi.NaElement('desired-attributes')
    storage_disk_info = netapp_utils.zapi.NaElement('storage-disk-info')
    disk_inv = netapp_utils.zapi.NaElement('disk-inventory-info')
    storage_disk_info.add_child_elem(disk_inv)
    desired_attributes.add_child_elem(storage_disk_info)
    disk_firmware_info_get.add_child_elem(desired_attributes)
    try:
        result = self.server.invoke_successfully(disk_firmware_info_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching disk module firmware  details: %s' % to_native(error), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        disk_info = result.get_child_by_name('attributes-list')
        disks = disk_info.get_children()
        for disk in disks:
            disk_id_fw_info[disk.get_child_content('disk-uid')] = disk.get_child_by_name('disk-inventory-info').get_child_content('firmware-revision')
    return disk_id_fw_info