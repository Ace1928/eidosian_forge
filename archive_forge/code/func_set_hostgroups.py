from __future__ import absolute_import, division, print_function
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.urls import open_url
import json
def set_hostgroups(self):
    hostgroup_list = self.call_url(self.url, self.url_username, self.url_password, url_path='/director/hostgroups')
    hostgroups = []
    for hostgroup in hostgroup_list['objects']:
        hostgroups.append(hostgroup['object_name'])
        self.inventory.add_group(hostgroup['object_name'])
    return hostgroups