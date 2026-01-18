from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def cifs_security_get_iter(self):
    """
        get current vserver cifs security.
        :return: a dict of vserver cifs security
        """
    cifs_security_get = netapp_utils.zapi.NaElement('cifs-security-get-iter')
    query = netapp_utils.zapi.NaElement('query')
    cifs_security = netapp_utils.zapi.NaElement('cifs-security')
    cifs_security.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(cifs_security)
    cifs_security_get.add_child_elem(query)
    cifs_security_details = dict()
    try:
        result = self.server.invoke_successfully(cifs_security_get, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error fetching cifs security from %s: %s' % (self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
    if result.get_child_by_name('num-records') and int(result.get_child_content('num-records')) > 0:
        cifs_security_info = result.get_child_by_name('attributes-list').get_child_by_name('cifs-security')
        for option, zapi_key in self.na_helper.zapi_int_keys.items():
            cifs_security_details[option] = self.na_helper.get_value_for_int(from_zapi=True, value=cifs_security_info.get_child_content(zapi_key))
        for option, zapi_key in self.na_helper.zapi_bool_keys.items():
            cifs_security_details[option] = self.na_helper.get_value_for_bool(from_zapi=True, value=cifs_security_info.get_child_content(zapi_key))
        for option, zapi_key in self.na_helper.zapi_str_keys.items():
            if cifs_security_info.get_child_content(zapi_key) is None:
                cifs_security_details[option] = None
            else:
                cifs_security_details[option] = cifs_security_info.get_child_content(zapi_key)
        return cifs_security_details
    return None