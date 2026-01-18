from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
@property
def host_exists(self):
    """Determine if the requested host exists
        As a side effect, set the full list of defined hosts in "all_hosts", and the target host in "host_obj".
        """
    match = False
    all_hosts = list()
    try:
        rc, all_hosts = self.request('storage-systems/%s/hosts' % self.ssid)
    except Exception as err:
        self.module.fail_json(msg='Failed to determine host existence. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
    for host in all_hosts:
        for port in host['hostSidePorts']:
            port['type'] = port['type'].lower()
            port['address'] = port['address'].lower()
        ports = dict(((port['label'], port['id']) for port in host['ports']))
        ports.update(dict(((port['label'], port['id']) for port in host['initiators'])))
        for host_side_port in host['hostSidePorts']:
            if host_side_port['label'] in ports:
                host_side_port['id'] = ports[host_side_port['label']]
        if host['label'].lower() == self.name.lower():
            self.host_obj = host
            match = True
    self.all_hosts = all_hosts
    return match