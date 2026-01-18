from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def create_subsystem(self):
    """
        Create a NVME Subsystem
        """
    if self.use_rest:
        return self.create_subsystem_rest()
    options = {'subsystem': self.parameters['subsystem'], 'ostype': self.parameters['ostype']}
    subsystem_create = netapp_utils.zapi.NaElement('nvme-subsystem-create')
    subsystem_create.translate_struct(options)
    try:
        self.server.invoke_successfully(subsystem_create, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating subsystem for %s: %s' % (self.parameters.get('subsystem'), to_native(error)), exception=traceback.format_exc())