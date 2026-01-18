from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
import ipaddress
def is_host_modified(self, host_details):
    """ Determines whether the Host details are to be updated or not """
    LOG.info('Checking host attribute values.')
    modified_flag = False
    if self.module.params['description'] is not None and self.module.params['description'] != host_details.description or (self.module.params['host_os'] is not None and self.module.params['host_os'] != host_details.os_type) or (self.module.params['new_host_name'] is not None and self.module.params['new_host_name'] != host_details.name) or (self.module.params['initiators'] is not None and self.module.params['initiators'] != self.get_host_initiators_list(host_details)):
        LOG.info('Modification required.')
        modified_flag = True
    return modified_flag