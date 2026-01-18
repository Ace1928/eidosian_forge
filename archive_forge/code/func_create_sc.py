from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.fusion.plugins.module_utils.fusion import (
from ansible_collections.purestorage.fusion.plugins.module_utils.parsing import (
from ansible_collections.purestorage.fusion.plugins.module_utils.startup import (
from ansible_collections.purestorage.fusion.plugins.module_utils.operations import (
def create_sc(module, fusion):
    """Create Storage Class"""
    sc_api_instance = purefusion.StorageClassesApi(fusion)
    if not module.params['size_limit']:
        module.params['size_limit'] = '4P'
    if not module.params['iops_limit']:
        module.params['iops_limit'] = '100000000'
    if not module.params['bw_limit']:
        module.params['bw_limit'] = '512G'
    size_limit = parse_number_with_metric_suffix(module, module.params['size_limit'])
    iops_limit = int(parse_number_with_metric_suffix(module, module.params['iops_limit'], factor=1000))
    bw_limit = parse_number_with_metric_suffix(module, module.params['bw_limit'])
    if bw_limit < 1048576 or bw_limit > 549755813888:
        module.fail_json(msg='Bandwidth limit is not within the required range')
    if iops_limit < 100 or iops_limit > 100000000:
        module.fail_json(msg='IOPs limit is not within the required range')
    if size_limit < 1048576 or size_limit > 4503599627370496:
        module.fail_json(msg='Size limit is not within the required range')
    changed = True
    id = None
    if not module.check_mode:
        if not module.params['display_name']:
            display_name = module.params['name']
        else:
            display_name = module.params['display_name']
        s_class = purefusion.StorageClassPost(name=module.params['name'], size_limit=size_limit, iops_limit=iops_limit, bandwidth_limit=bw_limit, display_name=display_name)
        op = sc_api_instance.create_storage_class(s_class, storage_service_name=module.params['storage_service'])
        res_op = await_operation(fusion, op)
        id = res_op.result.resource.id
    module.exit_json(changed=changed, id=id)