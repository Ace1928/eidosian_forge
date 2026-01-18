from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def modify_unix_group(self, params):
    """
        Modify an UNIX group from a vserver
        :param params: modify parameters
        :return: None
        """
    if 'users' in params:
        self.modify_users_in_group()
        if len(params) == 1:
            return
    group_modify = netapp_utils.zapi.NaElement('name-mapping-unix-group-modify')
    group_details = {'group-name': self.parameters['name']}
    for key in params:
        if key in self.na_helper.zapi_int_keys:
            zapi_key = self.na_helper.zapi_int_keys.get(key)
            group_details[zapi_key] = self.na_helper.get_value_for_int(from_zapi=True, value=params[key])
    group_modify.translate_struct(group_details)
    try:
        self.server.invoke_successfully(group_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error modifying UNIX group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())