from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def apply_target_changes(self):
    update = False
    target = self.target
    body = dict()
    if self.ping != target['ping']:
        update = True
        body['icmpPingResponseEnabled'] = self.ping
    if self.unnamed_discovery != target['unnamed_discovery']:
        update = True
        body['unnamedDiscoverySessionsEnabled'] = self.unnamed_discovery
    if update and (not self.check_mode):
        try:
            self.request('storage-systems/%s/iscsi/entity' % self.ssid, method='POST', data=body)
        except Exception as err:
            self.module.fail_json(msg='Failed to update the iSCSI target settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    return update