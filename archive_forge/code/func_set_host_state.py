from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
from time import sleep
def set_host_state(self):
    """Checking and applying ESXi host configuration one by one,
        from prepared list of hosts in `self.hosts`.
        For every host applied:
        - user input checking done via calling `sanitize_params` method
        - checks hardware compatibility with user input `check_compatibility`
        - conf changes created via `make_diff`
        - changes applied via calling `_update_sriov` method
        - host state before and after via calling `_check_sriov`
        """
    self.sanitize_params()
    change_list = []
    changed = False
    for host in self.hosts:
        self.results['before'][host.name] = {}
        self.results['after'][host.name] = {}
        self.results['changes'][host.name] = {}
        self.results['before'][host.name] = self._check_sriov(host)
        self.check_compatibility(self.results['before'][host.name], host.name)
        diff = self.make_diff(self.results['before'][host.name], host.name)
        self.results['changes'][host.name] = diff
        if not diff['change']:
            change_list.append(False)
            self.results['after'][host.name] = self._check_sriov(host)
            if self.results['before'][host.name]['rebootRequired'] != self.results['after'][host.name]['rebootRequired']:
                self.results['changes'][host.name]['rebootRequired'] = self.results['after'][host.name]['rebootRequired']
            continue
        success = self._update_sriov(host, self.sriov_on, self.num_virt_func)
        if success:
            change_list.append(True)
        else:
            change_list.append(False)
        self.results['after'][host.name] = self._check_sriov(host)
        self.results['changes'][host.name].update({'rebootRequired': self.results['after'][host.name]['rebootRequired']})
    if any(change_list):
        changed = True
    self.module.exit_json(changed=changed, diff=self.results)