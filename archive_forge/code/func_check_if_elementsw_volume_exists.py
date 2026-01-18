from __future__ import absolute_import, division, print_function
import re
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def check_if_elementsw_volume_exists(self, path, elementsw_helper):
    """
        Check if remote ElementSW volume exists
        :return: None
        """
    volume_id, vol_id = (None, path.split('/')[-1])
    try:
        volume_id = elementsw_helper.volume_id_exists(int(vol_id))
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error fetching Volume details', exception=to_native(err))
    if volume_id is None:
        self.module.fail_json(msg='Error: Source volume does not exist in the ElementSW cluster')