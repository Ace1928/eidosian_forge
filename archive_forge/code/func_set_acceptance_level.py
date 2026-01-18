from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def set_acceptance_level(self):
    change = []
    for host in self.hosts:
        self.hosts_facts[host.name] = dict(level='', error='NA')
        host_image_config_mgr = host.configManager.imageConfigManager
        if host_image_config_mgr:
            try:
                self.hosts_facts[host.name]['level'] = host_image_config_mgr.HostImageConfigGetAcceptance()
            except vim.fault.HostConfigFault as e:
                self.hosts_facts[host.name]['error'] = to_native(e.msg)
        host_changed = False
        if self.hosts_facts[host.name]['level'] != self.desired_state:
            try:
                if self.module.check_mode:
                    self.hosts_facts[host.name]['level'] = self.desired_state
                else:
                    host_image_config_mgr.UpdateHostImageAcceptanceLevel(newAcceptanceLevel=self.desired_state)
                    self.hosts_facts[host.name]['level'] = host_image_config_mgr.HostImageConfigGetAcceptance()
                host_changed = True
            except vim.fault.HostConfigFault as e:
                self.hosts_facts[host.name]['error'] = to_native(e.msg)
        change.append(host_changed)
    self.module.exit_json(changed=any(change), facts=self.hosts_facts)