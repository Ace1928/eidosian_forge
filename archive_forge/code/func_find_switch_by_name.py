from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
def find_switch_by_name(module):
    """ Find switch by name """
    switch = module.params['switch_name']
    path = f'fc/switches?name={switch}'
    system = get_system(module)
    try:
        switch_result = system.api.get(path=path).get_result()
        if not switch_result:
            msg = f'Cannot find switch {switch}'
            module.exit_json(msg=msg)
    except APICommandFailed as err:
        msg = f'Cannot find switch {switch}: {err}'
        module.exit_json(msg=msg)
    return switch_result[0]