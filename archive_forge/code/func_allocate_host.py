from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.general.plugins.module_utils.opennebula import OpenNebulaModule
def allocate_host(self):
    """
        Creates a host entry in OpenNebula
        Returns: True on success, fails otherwise.

        """
    if not self.one.host.allocate(self.get_parameter('name'), self.get_parameter('vmm_mad_name'), self.get_parameter('im_mad_name'), self.get_parameter('cluster_id')):
        self.fail(msg='could not allocate host')
    else:
        self.result['changed'] = True
    return True