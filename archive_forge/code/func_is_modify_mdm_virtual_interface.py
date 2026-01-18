from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
import copy
def is_modify_mdm_virtual_interface(virtual_ip_interfaces, clear_interfaces, mdm_details):
    """Check if modification in MDM virtual IP interface required."""
    modify_list = []
    clear = False
    existing_interfaces = mdm_details['virtualInterfaces']
    if clear_interfaces is None and len(existing_interfaces) == len(virtual_ip_interfaces) and (set(existing_interfaces) == set(virtual_ip_interfaces)):
        LOG.info('No changes required for virtual IP interface.')
        return (None, False)
    if clear_interfaces and len(mdm_details['virtualInterfaces']) == 0:
        LOG.info('No change required for clear interface.')
        return (None, False)
    elif clear_interfaces and len(mdm_details['virtualInterfaces']) != 0 and (virtual_ip_interfaces is None):
        LOG.info('Clear all interfaces of the MDM.')
        clear = True
        return (None, clear)
    if virtual_ip_interfaces and clear_interfaces is None:
        for interface in virtual_ip_interfaces:
            modify_list.append(interface)
        return (modify_list, clear)