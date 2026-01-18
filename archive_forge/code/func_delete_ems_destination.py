from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_ems_destination(self, name):
    api = 'support/ems/destinations'
    dummy, error = rest_generic.delete_async(self.rest_api, api, name)
    self.fail_on_error(error, 'deleting EMS destination for %s' % name)