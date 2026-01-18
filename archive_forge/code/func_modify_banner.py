from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def modify_banner(self, modify):
    login_banner_modify = netapp_utils.zapi.NaElement('vserver-login-banner-modify-iter')
    login_banner_modify.add_new_child('message', modify['banner'])
    query = netapp_utils.zapi.NaElement('query')
    login_banner_info = netapp_utils.zapi.NaElement('vserver-login-banner-info')
    login_banner_info.add_new_child('vserver', self.parameters['vserver'])
    query.add_child_elem(login_banner_info)
    login_banner_modify.add_child_elem(query)
    try:
        self.server.invoke_successfully(login_banner_modify, enable_tunneling=False)
    except netapp_utils.zapi.NaApiError as err:
        self.module.fail_json(msg='Error modifying login_banner: %s' % to_native(err), exception=traceback.format_exc())