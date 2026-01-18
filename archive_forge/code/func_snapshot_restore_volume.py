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
def snapshot_restore_volume(self):
    if self.use_rest:
        return self.snapshot_restore_volume_rest()
    snapshot_restore = netapp_utils.zapi.NaElement.create_node_with_children('snapshot-restore-volume', **{'snapshot': self.parameters['snapshot_restore'], 'volume': self.parameters['name']})
    if self.parameters.get('force_restore') is not None:
        snapshot_restore.add_new_child('force', str(self.parameters['force_restore']))
    if self.parameters.get('preserve_lun_ids') is not None:
        snapshot_restore.add_new_child('preserve-lun-ids', str(self.parameters['preserve_lun_ids']))
    try:
        self.server.invoke_successfully(snapshot_restore, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error restoring volume %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())