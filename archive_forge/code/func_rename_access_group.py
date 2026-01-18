from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def rename_access_group(self):
    """
        Rename the Access Group to the new name
        """
    try:
        self.sfe.modify_volume_access_group(volume_access_group_id=self.from_group_id, virtual_network_id=self.virtual_network_id, virtual_network_tags=self.virtual_network_tags, name=self.access_group_name, initiators=self.initiators, volumes=self.volumes, attributes=self.attributes)
    except Exception as e:
        self.module.fail_json(msg='Error updating volume access group %s: %s' % (self.from_name, to_native(e)), exception=traceback.format_exc())