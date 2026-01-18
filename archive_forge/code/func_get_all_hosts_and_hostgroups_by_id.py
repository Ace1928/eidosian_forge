from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
def get_all_hosts_and_hostgroups_by_id(self):
    """Retrieve and return a dictionary of all host and host groups keyed by name."""
    if not self.cache['get_all_hosts_and_hostgroups_by_id']:
        try:
            rc, hostgroups = self.request('storage-systems/%s/host-groups' % self.ssid)
            hostgroup_by_id = dict(((hostgroup['id'], hostgroup) for hostgroup in hostgroups))
            rc, hosts = self.request('storage-systems/%s/hosts' % self.ssid)
            for host in hosts:
                if host['clusterRef'] != '0000000000000000000000000000000000000000':
                    hostgroup_name = hostgroup_by_id[host['clusterRef']]['name']
                    if host['clusterRef'] not in self.cache['get_all_hosts_and_hostgroups_by_id'].keys():
                        hostgroup_by_id[host['clusterRef']].update({'hostgroup': True, 'host_ids': [host['id']]})
                        self.cache['get_all_hosts_and_hostgroups_by_id'].update({host['clusterRef']: hostgroup_by_id[host['clusterRef']]})
                        self.cache['get_all_hosts_and_hostgroups_by_name'].update({hostgroup_name: hostgroup_by_id[host['clusterRef']]})
                    else:
                        self.cache['get_all_hosts_and_hostgroups_by_id'][host['clusterRef']]['host_ids'].append(host['id'])
                        self.cache['get_all_hosts_and_hostgroups_by_name'][hostgroup_name]['host_ids'].append(host['id'])
                self.cache['get_all_hosts_and_hostgroups_by_id'].update({host['id']: host, 'hostgroup': False})
                self.cache['get_all_hosts_and_hostgroups_by_name'].update({host['name']: host, 'hostgroup': False})
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve all host and host group objects! Error [%s]. Array [%s].' % (error, self.ssid))
    return self.cache['get_all_hosts_and_hostgroups_by_id']