from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def check_host_state(self):
    """Check ESXi host configuration"""
    change_list = []
    changed = False
    for host in self.hosts:
        self.results[host.name] = dict()
        if host.runtime.connectionState == 'connected':
            ntp_servers_configured, ntp_servers_to_change = self.check_ntp_servers(host=host)
            if self.desired_state:
                self.results[host.name]['state'] = self.desired_state
                if ntp_servers_to_change:
                    self.results[host.name]['ntp_servers_changed'] = ntp_servers_to_change
                    operation = 'add' if self.desired_state == 'present' else 'delete'
                    new_ntp_servers = self.update_ntp_servers(host=host, ntp_servers_configured=ntp_servers_configured, ntp_servers_to_change=ntp_servers_to_change, operation=operation)
                    self.results[host.name]['ntp_servers_current'] = new_ntp_servers
                    self.results[host.name]['changed'] = True
                    change_list.append(True)
                else:
                    self.results[host.name]['ntp_servers_current'] = ntp_servers_configured
                    if self.verbose:
                        self.results[host.name]['msg'] = 'NTP servers already added' if self.desired_state == 'present' else 'NTP servers already removed'
                    self.results[host.name]['changed'] = False
                    change_list.append(False)
            else:
                self.results[host.name]['ntp_servers'] = self.ntp_servers
                if ntp_servers_to_change:
                    self.results[host.name]['ntp_servers_changed'] = self.get_differt_entries(ntp_servers_configured, ntp_servers_to_change)
                    self.update_ntp_servers(host=host, ntp_servers_configured=ntp_servers_configured, ntp_servers_to_change=ntp_servers_to_change, operation='overwrite')
                    self.results[host.name]['changed'] = True
                    change_list.append(True)
                else:
                    if self.verbose:
                        self.results[host.name]['msg'] = 'NTP servers already configured'
                    self.results[host.name]['changed'] = False
                    change_list.append(False)
        else:
            self.results[host.name]['changed'] = False
            self.results[host.name]['msg'] = 'Host %s is disconnected and cannot be changed.' % host.name
    if any(change_list):
        changed = True
    self.module.exit_json(changed=changed, host_ntp_status=self.results)