from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
@property
def current_hosts_in_host_group(self):
    """Retrieve the current hosts associated with the current hostgroup."""
    current_hosts = []
    for group in self.host_groups:
        if group['name'] == self.name:
            current_hosts = group['hosts']
            break
    return current_hosts