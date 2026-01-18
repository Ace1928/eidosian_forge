from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, get_all_objs, wait_for_task, PyVmomi
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
def find_host_system(self):
    if self.esxi_hostname:
        host_system_obj = self.find_hostsystem_by_name(host_name=self.esxi_hostname)
        if host_system_obj:
            return host_system_obj
        else:
            self.module.fail_json(msg='Failed to find ESXi %s' % self.esxi_hostname)
    host_system = get_all_objs(self.content, [vim.HostSystem])
    return list(host_system)[0]