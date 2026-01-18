from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def delete_subsystem(self):
    """
        Delete a NVME subsystem
        """
    if self.use_rest:
        return self.delete_subsystem_rest()
    options = {'subsystem': self.parameters['subsystem'], 'skip-host-check': 'true' if self.parameters.get('skip_host_check') else 'false', 'skip-mapped-check': 'true' if self.parameters.get('skip_mapped_check') else 'false'}
    subsystem_delete = netapp_utils.zapi.NaElement.create_node_with_children('nvme-subsystem-delete', **options)
    try:
        self.server.invoke_successfully(subsystem_delete, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error deleting subsystem for %s: %s' % (self.parameters.get('subsystem'), to_native(error)), exception=traceback.format_exc())