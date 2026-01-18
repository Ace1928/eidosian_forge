from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
def remove_datacenter(module, profitbricks):
    """
    Removes a Datacenter.

    This will remove a datacenter.

    module : AnsibleModule object
    profitbricks: authenticated profitbricks object.

    Returns:
        True if the datacenter was deleted, false otherwise
    """
    name = module.params.get('name')
    changed = False
    if uuid_match.match(name):
        _remove_datacenter(module, profitbricks, name)
        changed = True
    else:
        datacenters = profitbricks.list_datacenters()
        for d in datacenters['items']:
            vdc = profitbricks.get_datacenter(d['id'])
            if name == vdc['properties']['name']:
                name = d['id']
                _remove_datacenter(module, profitbricks, name)
                changed = True
    return changed