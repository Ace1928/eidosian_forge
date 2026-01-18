from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_unix_group(self):
    """
        Creates an UNIX group in the specified Vserver

        :return: None
        """
    if self.parameters.get('id') is None:
        self.module.fail_json(msg='Error: Missing a required parameter for create: (id)')
    group_create = netapp_utils.zapi.NaElement('name-mapping-unix-group-create')
    group_details = {}
    for item in self.parameters:
        if item in self.na_helper.zapi_string_keys:
            zapi_key = self.na_helper.zapi_string_keys.get(item)
            group_details[zapi_key] = self.parameters[item]
        elif item in self.na_helper.zapi_bool_keys:
            zapi_key = self.na_helper.zapi_bool_keys.get(item)
            group_details[zapi_key] = self.na_helper.get_value_for_bool(from_zapi=False, value=self.parameters[item])
        elif item in self.na_helper.zapi_int_keys:
            zapi_key = self.na_helper.zapi_int_keys.get(item)
            group_details[zapi_key] = self.na_helper.get_value_for_int(from_zapi=True, value=self.parameters[item])
    group_create.translate_struct(group_details)
    try:
        self.server.invoke_successfully(group_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating UNIX group %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())
    if self.parameters.get('users') is not None:
        self.modify_users_in_group()