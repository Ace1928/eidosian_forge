from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_service_policy(self, current):
    api = 'network/ip/service-policies/%s' % current['uuid']
    dummy, error = rest_generic.delete_async(self.rest_api, api, None, None)
    if error:
        msg = 'Error in delete_service_policy: %s' % error
        self.module.fail_json(msg=msg)