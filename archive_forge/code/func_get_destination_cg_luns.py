from __future__ import absolute_import, division, print_function
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def get_destination_cg_luns(self, source_lun_list):
    """ Form destination cg lun list """
    destination_cg_lun_list = []
    if source_lun_list is not None:
        for source_lun in source_lun_list:
            destination_cg_lun_info = utils.UnityStorageResource()
            destination_cg_lun_info.name = 'DR_' + source_lun.name
            destination_cg_lun_info.is_thin_enabled = source_lun.is_thin_enabled
            destination_cg_lun_info.size_total = source_lun.size_total
            destination_cg_lun_info.id = source_lun.id
            destination_cg_lun_info.is_data_reduction_enabled = source_lun.is_data_reduction_enabled
            destination_cg_lun_list.append(destination_cg_lun_info)
    return destination_cg_lun_list