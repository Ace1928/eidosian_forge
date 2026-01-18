from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
def expand_volume(self):
    """Expand the storage specifications for the existing thick/thin volume.

        :raise AnsibleFailJson when a thick/thin volume expansion request fails.
        """
    request_body = self.get_expand_volume_changes()
    if request_body:
        if self.volume_detail['thinProvisioned']:
            try:
                rc, resp = self.request('storage-systems/%s/thin-volumes/%s/expand' % (self.ssid, self.volume_detail['id']), data=request_body, method='POST')
            except Exception as err:
                self.module.fail_json(msg='Failed to expand thin volume. Volume [%s]. Array Id [%s]. Error[%s].' % (self.name, self.ssid, to_native(err)))
            self.module.log('Thin volume specifications have been expanded.')
        else:
            try:
                rc, resp = self.request('storage-systems/%s/volumes/%s/expand' % (self.ssid, self.volume_detail['id']), data=request_body, method='POST')
            except Exception as err:
                self.module.fail_json(msg='Failed to expand volume.  Volume [%s].  Array Id [%s]. Error[%s].' % (self.name, self.ssid, to_native(err)))
            self.module.log('Volume storage capacities have been expanded.')