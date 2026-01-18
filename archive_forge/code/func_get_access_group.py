from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_access_group(self, name):
    """
        Get Access Group
            :description: Get Access Group object for a given name

            :return: object (Group object)
            :rtype: object (Group object)
        """
    access_groups_list = self.sfe.list_volume_access_groups()
    group_obj = None
    for group in access_groups_list.volume_access_groups:
        if str(group.volume_access_group_id) == name:
            group_obj = group
        elif group.name == name:
            group_obj = group
    return group_obj