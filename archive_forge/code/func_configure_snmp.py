from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def configure_snmp(self, actual_networks, actual_usm_users):
    """
        Configure snmp
        """
    try:
        self.sfe.set_snmp_acl(networks=[actual_networks], usm_users=[actual_usm_users])
    except Exception as exception_object:
        self.module.fail_json(msg='Error Configuring snmp feature %s' % to_native(exception_object), exception=traceback.format_exc())