from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def set_snapshot_auto_delete(self):
    options = {'volume': self.parameters['name']}
    desired_options = self.parameters['snapshot_auto_delete']
    for key, value in desired_options.items():
        options['option-name'] = key
        options['option-value'] = str(value)
        snapshot_auto_delete = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-autodelete-set-option', **options)
        try:
            self.server.invoke_successfully(snapshot_auto_delete, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.wrap_fail_json(msg='Error setting snapshot auto delete options for volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())