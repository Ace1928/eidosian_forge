from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def get_metadata_vol(module, disable_fail):
    """ Get metadata about a volume """
    system = get_system(module)
    object_type = module.params['object_type']
    object_name = module.params['object_name']
    key = module.params['key']
    metadata = None
    vol = get_volume(module, system)
    if vol:
        path = f'metadata/{vol.id}/{key}'
        try:
            metadata = system.api.get(path=path)
        except APICommandFailed:
            if not disable_fail:
                module.fail_json(f'Cannot find {object_type} metadata key. Volume {object_name} key {key} not found')
    elif not disable_fail:
        msg = f'Volume with object name {object_name} not found. Cannot stat its metadata.'
        module.fail_json(msg=msg)
    return metadata