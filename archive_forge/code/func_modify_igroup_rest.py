from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_igroup_rest(self, uuid, modify):
    api = 'protocols/san/igroups'
    body = dict()
    for option in modify:
        if option not in self.rest_modify_zapi_to_rest:
            self.module.fail_json(msg='Error: modifying %s is not supported in REST' % option)
        body[self.rest_modify_zapi_to_rest[option]] = modify[option]
    if body:
        dummy, error = rest_generic.patch_async(self.rest_api, api, uuid, body)
        self.fail_on_error(error)