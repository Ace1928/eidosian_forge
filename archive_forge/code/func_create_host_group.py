from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
def create_host_group(self):
    """Create host group."""
    data = {'name': self.name, 'hosts': self.hosts}
    response = None
    try:
        rc, response = self.request('storage-systems/%s/host-groups' % self.ssid, method='POST', data=data)
    except Exception as error:
        self.module.fail_json(msg='Failed to create host group. Array id [%s]. Error[%s].' % (self.ssid, to_native(error)))
    return response