from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def find_session_by_name(self):
    """Finds a session by name
        Returns
        -------
        vim.dvs.VmwareDistributedVirtualSwitch.VspanSession
            The session if there was a session by the given name, else returns None
        """
    for vspan_session in self.dv_switch.config.vspanSession:
        if vspan_session.name == self.name:
            return vspan_session
    return None