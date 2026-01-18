from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
def cifs_security_modify(self, modify):
    """
        :param modify: A list of attributes to modify
        :return: None
        """
    cifs_security_modify = netapp_utils.zapi.NaElement('cifs-security-modify')
    for attribute in modify:
        cifs_security_modify.add_new_child(self.attribute_to_name(attribute), str(self.parameters[attribute]))
    try:
        self.server.invoke_successfully(cifs_security_modify, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as e:
        self.module.fail_json(msg='Error modifying cifs security on %s: %s' % (self.parameters['vserver'], to_native(e)), exception=traceback.format_exc())