from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.vexata import (
def get_ig_id(module, array):
    """Retrieve a named ig's id if it exists, error if absent."""
    name = module.params['ig']
    try:
        igs = array.list_igs()
        ig = filter(lambda ig: ig['name'] == name, igs)
        if len(ig) == 1:
            return ig[0]['id']
        else:
            module.fail_json(msg='Initiator group {0} was not found.'.format(name))
    except Exception:
        module.fail_json(msg='Error while attempting to retrieve initiator groups.')