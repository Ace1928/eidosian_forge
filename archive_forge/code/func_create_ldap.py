from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def create_ldap(self):
    """
        Create LDAP configuration
        """
    options = {'client-config': self.parameters['name'], 'client-enabled': 'true'}
    if self.parameters.get('skip_config_validation') is not None:
        options['skip-config-validation'] = self.parameters['skip_config_validation']
    ldap_create = netapp_utils.zapi.NaElement.create_node_with_children('ldap-config-create', **options)
    try:
        self.server.invoke_successfully(ldap_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as errcatch:
        self.module.fail_json(msg='Error creating LDAP configuration %s: %s' % (self.parameters['name'], to_native(errcatch)), exception=traceback.format_exc())