from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
@property
def host_groups(self):
    """Retrieve a list of existing host groups."""
    host_groups = []
    hosts = []
    try:
        rc, host_groups = self.request('storage-systems/%s/host-groups' % self.ssid)
        rc, hosts = self.request('storage-systems/%s/hosts' % self.ssid)
    except Exception as error:
        self.module.fail_json(msg='Failed to retrieve host group information. Array id [%s].  Error[%s].' % (self.ssid, to_native(error)))
    host_groups = [{'id': group['clusterRef'], 'name': group['name']} for group in host_groups]
    for group in host_groups:
        hosts_ids = []
        for host in hosts:
            if group['id'] == host['clusterRef']:
                hosts_ids.append(host['hostRef'])
        group.update({'hosts': hosts_ids})
    return host_groups