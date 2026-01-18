from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_aggr_object_store(self, body, uuid=None):
    """
        modify aggregate object store config
        :return: None
        """
    api = 'cloud/targets'
    dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
    if error:
        self.module.fail_json(msg='Error %s' % error)