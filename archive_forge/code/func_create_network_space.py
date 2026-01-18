from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def create_network_space(module, system):
    """ Create a network space """
    if not module.check_mode:
        create_empty_network_space(module, system)
        space_id = find_network_space_id(module, system)
        add_ips_to_network_space(module, system, space_id)
        changed = True
    else:
        changed = False
    return changed