from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_scanner_pool(self, modify):
    """
        Modify a scanner pool
        :return: nothing
        """
    if self.use_rest:
        return self.modify_scanner_pool_rest(modify)
    vscan_pool_modify = netapp_utils.zapi.NaElement('vscan-scanner-pool-modify')
    vscan_pool_modify.add_new_child('scanner-pool', self.parameters['scanner_pool'])
    for key in modify:
        if key == 'privileged_users':
            users_obj = netapp_utils.zapi.NaElement('privileged-users')
            vscan_pool_modify.add_child_elem(users_obj)
            for user in modify['privileged_users']:
                users_obj.add_new_child('privileged-user', user)
        elif key == 'hostnames':
            string_obj = netapp_utils.zapi.NaElement('hostnames')
            vscan_pool_modify.add_child_elem(string_obj)
            for hostname in modify['hostnames']:
                string_obj.add_new_child('string', hostname)
        elif key != 'scanner_policy':
            vscan_pool_modify.add_new_child(self.attribute_to_name(key), str(modify[key]))
    try:
        self.server.invoke_successfully(vscan_pool_modify, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying Vscan Scanner Pool %s: %s' % (self.parameters['scanner_pool'], to_native(error)), exception=traceback.format_exc())