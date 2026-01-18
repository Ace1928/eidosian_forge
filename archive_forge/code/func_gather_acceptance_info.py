from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from ansible.module_utils._text import to_native
def gather_acceptance_info(self):
    for host in self.hosts:
        self.hosts_facts[host.name] = dict(level='', error='NA')
        host_image_config_mgr = host.configManager.imageConfigManager
        if host_image_config_mgr:
            try:
                self.hosts_facts[host.name]['level'] = host_image_config_mgr.HostImageConfigGetAcceptance()
            except vim.fault.HostConfigFault as e:
                self.hosts_facts[host.name]['error'] = to_native(e.msg)
    self.module.exit_json(changed=False, host_acceptance_info=self.hosts_facts)