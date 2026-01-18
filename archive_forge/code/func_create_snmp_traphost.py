from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_snmp_traphost(self):
    api = 'support/snmp/traphosts'
    params = {'host': self.parameters.get('host')}
    dummy, error = rest_generic.post_async(self.rest_api, api, params)
    if error:
        self.module.fail_json(msg='Error creating traphost: %s' % error)