from __future__ import (absolute_import, division, print_function)
from ansible.plugins.lookup import LookupBase
from ansible.errors import AnsibleError
def add_volume_mapping(self, name, host, lun):
    """Add volume mapping to record table (luns_by_target)."""
    for host_group in self.array_facts['netapp_host_groups']:
        if host == host_group['name']:
            self.luns_by_target[host].append([name, lun])
            for hostgroup_host in host_group['hosts']:
                self.luns_by_target[hostgroup_host].append([name, lun])
            break
    else:
        self.luns_by_target[host].append([name, lun])