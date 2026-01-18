from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def get_vvols_status(self):
    """
        get vvols status
        """
    feature_status = self.sfe.get_feature_status(feature='vvols')
    if feature_status is not None:
        return feature_status.features[0].enabled
    return None