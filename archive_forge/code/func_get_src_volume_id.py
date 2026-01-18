from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
def get_src_volume_id(self):
    """
            Return volume id if found
        """
    src_vol_id = self.elementsw_helper.volume_exists(self.src_volume_id, self.account_id)
    if src_vol_id is not None:
        self.src_volume_id = src_vol_id
        return self.src_volume_id
    return None