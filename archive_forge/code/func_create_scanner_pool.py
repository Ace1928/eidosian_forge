from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def create_scanner_pool(self):
    """
        Create a Vscan Scanner Pool
        :return: nothing
        """
    if self.use_rest:
        return self.create_scanner_pool_rest()
    scanner_pool_obj = netapp_utils.zapi.NaElement('vscan-scanner-pool-create')
    if self.parameters['hostnames']:
        string_obj = netapp_utils.zapi.NaElement('hostnames')
        scanner_pool_obj.add_child_elem(string_obj)
        for hostname in self.parameters['hostnames']:
            string_obj.add_new_child('string', hostname)
    if self.parameters['privileged_users']:
        users_obj = netapp_utils.zapi.NaElement('privileged-users')
        scanner_pool_obj.add_child_elem(users_obj)
        for user in self.parameters['privileged_users']:
            users_obj.add_new_child('privileged-user', user)
    scanner_pool_obj.add_new_child('scanner-pool', self.parameters['scanner_pool'])
    try:
        self.server.invoke_successfully(scanner_pool_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating Vscan Scanner Pool %s: %s' % (self.parameters['scanner_policy'], to_native(error)), exception=traceback.format_exc())