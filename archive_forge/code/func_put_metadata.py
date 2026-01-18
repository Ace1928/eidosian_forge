from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.infinidat.infinibox.plugins.module_utils.infinibox import (
@api_wrapper
def put_metadata(module):
    """Create metadata key with a value.  The changed variable is found elsewhere."""
    system = get_system(module)
    object_type = module.params['object_type']
    key = module.params['key']
    value = module.params['value']
    if object_type == 'system':
        path = 'metadata/system'
    elif object_type == 'vol':
        vol = get_volume(module, system)
        if not vol:
            object_name = module.params['object_name']
            msg = f'Volume {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{vol.id}'
    elif object_type == 'fs':
        fs = get_filesystem(module, system)
        if not fs:
            object_name = module.params['object_name']
            msg = f'File system {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{fs.id}'
    elif object_type == 'host':
        host = get_host(module, system)
        if not host:
            object_name = module.params['object_name']
            msg = f'Cluster {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{host.id}'
    elif object_type == 'cluster':
        cluster = get_cluster(module, system)
        if not cluster:
            object_name = module.params['object_name']
            msg = f'Cluster {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{cluster.id}'
    elif object_type == 'fs-snap':
        fssnap = get_filesystem(module, system)
        if not fssnap:
            object_name = module.params['object_name']
            msg = f'File system snapshot {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{fssnap.id}'
    elif object_type == 'pool':
        pool = get_pool(module, system)
        if not pool:
            object_name = module.params['object_name']
            msg = f'Pool {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{pool.id}'
    elif object_type == 'vol-snap':
        volsnap = get_volume(module, system)
        if not volsnap:
            object_name = module.params['object_name']
            msg = f'Volume snapshot {object_name} not found. Cannot add metadata key {key}.'
            module.fail_json(msg=msg)
        path = f'metadata/{volsnap.id}'
    data = {key: value}
    system.api.put(path=path, data=data)