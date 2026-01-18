from __future__ import absolute_import, division, print_function
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
def apply_iscsi_settings(self):
    """Update the iSCSI target alias and CHAP settings"""
    update = False
    target = self.target
    body = dict()
    if self.name is not None and self.name != target['alias']:
        update = True
        body['alias'] = self.name
    if self.chap_secret:
        update = True
        body.update(dict(enableChapAuthentication=True, chapSecret=self.chap_secret))
    elif target['chap']:
        update = True
        body.update(dict(enableChapAuthentication=False))
    if update and (not self.check_mode):
        try:
            self.request('storage-systems/%s/iscsi/target-settings' % self.ssid, method='POST', data=body)
        except Exception as err:
            self.module.fail_json(msg='Failed to update the iSCSI target settings. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    return update