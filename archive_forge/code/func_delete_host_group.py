from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def delete_host_group(self, unassign_hosts=True):
    """Delete host group"""
    if unassign_hosts:
        self.unassign_hosts()
    try:
        rc, resp = self.request('storage-systems/%s/host-groups/%s' % (self.ssid, self.current_host_group['id']), method='DELETE')
    except Exception as error:
        self.module.fail_json(msg='Failed to delete host group. Array id [%s]. Error[%s].' % (self.ssid, to_native(error)))