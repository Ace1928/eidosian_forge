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
def change_volume_state(self, call_from_delete_vol=False):
    """
        Change volume's state (offline/online).
        """
    if self.use_rest:
        return self.change_volume_state_rest()
    if self.parameters['is_online'] and (not call_from_delete_vol):
        vol_state_zapi, vol_name_zapi, action = ['volume-online-async', 'volume-name', 'online'] if self.parameters['is_infinite'] or self.volume_style == 'flexgroup' else ['volume-online', 'name', 'online']
    else:
        vol_state_zapi, vol_name_zapi, action = ['volume-offline-async', 'volume-name', 'offline'] if self.parameters['is_infinite'] or self.volume_style == 'flexgroup' else ['volume-offline', 'name', 'offline']
        volume_unmount = netapp_utils.zapi.NaElement.create_node_with_children('volume-unmount', **{'volume-name': self.parameters['name']})
    volume_change_state = netapp_utils.zapi.NaElement.create_node_with_children(vol_state_zapi, **{vol_name_zapi: self.parameters['name']})
    errors = []
    if not self.parameters['is_online'] or call_from_delete_vol:
        try:
            self.server.invoke_successfully(volume_unmount, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            errors.append('Error unmounting volume %s: %s' % (self.parameters['name'], to_native(error)))
    state = 'online' if self.parameters['is_online'] and (not call_from_delete_vol) else 'offline'
    try:
        result = self.server.invoke_successfully(volume_change_state, enable_tunneling=True)
        if self.volume_style == 'flexgroup' or self.parameters['is_infinite']:
            self.check_invoke_result(result, action)
    except netapp_utils.zapi.NaApiError as error:
        errors.append('Error changing the state of volume %s to %s: %s' % (self.parameters['name'], state, to_native(error)))
    if errors and (not call_from_delete_vol):
        self.module.fail_json(msg=', '.join(errors), exception=traceback.format_exc())
    return (state, errors)