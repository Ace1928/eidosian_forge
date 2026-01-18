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
def check_if_remote_volume_exists_rest(self):
    """
        Check the remote volume exists using REST
        """
    if self.src_use_rest:
        if self.parameters.get('source_volume') is not None and self.parameters.get('source_vserver') is not None:
            volume_name = self.parameters['source_volume']
            svm_name = self.parameters['source_vserver']
            options = {'name': volume_name, 'svm.name': svm_name, 'fields': 'name,svm.name'}
            api = 'storage/volumes'
            record, error = rest_generic.get_one_record(self.src_rest_api, api, options)
            if error:
                self.module.fail_json(msg='Error fetching source volume: %s' % error)
            return record is not None
        return False
    self.module.fail_json(msg='REST is not supported on Source')