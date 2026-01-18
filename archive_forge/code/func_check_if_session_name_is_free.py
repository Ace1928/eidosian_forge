from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def check_if_session_name_is_free(self):
    """Checks whether the name is used or not
        Returns
        -------
        bool
            True if the name is free and False if it is used.
        """
    for vspan_session in self.dv_switch.config.vspanSession:
        if vspan_session.name == self.name:
            return False
    return True