from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def create_access_group(self):
    """
        Create the Access Group
        """
    try:
        self.sfe.create_volume_access_group(name=self.access_group_name, initiators=self.initiators, volumes=self.volumes, virtual_network_id=self.virtual_network_id, virtual_network_tags=self.virtual_network_tags, attributes=self.attributes)
    except Exception as e:
        self.module.fail_json(msg='Error creating volume access group %s: %s' % (self.access_group_name, to_native(e)), exception=traceback.format_exc())