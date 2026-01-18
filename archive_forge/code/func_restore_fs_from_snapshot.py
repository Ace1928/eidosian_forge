from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def restore_fs_from_snapshot(module, system):
    """ Use snapshot to restore a file system """
    changed = False
    is_restoring = module.params['restore_fs_from_snapshot']
    fs_type = module.params['fs_type']
    snap_name = module.params['name']
    snap_id = find_fs_id(module, system, snap_name)
    parent_fs_name = module.params['parent_fs_name']
    parent_fs_id = find_fs_id(module, system, parent_fs_name)
    if not is_restoring:
        raise AssertionError('A programming error occurred. is_restoring is not True')
    if fs_type != 'snapshot':
        module.exit_json(msg="Cannot restore a parent file system from snapshot unless the file system type is 'snapshot'")
    if not parent_fs_name:
        module.exit_json(msg='Cannot restore a parent file system from snapshot unless the parent file system name is specified')
    if not module.check_mode:
        restore_url = f'filesystems/{parent_fs_id}/restore?approved=true'
        restore_data = {'source_id': snap_id}
        try:
            system.api.post(path=restore_url, data=restore_data)
            changed = True
        except APICommandFailed as err:
            module.fail_json(msg=f'Cannot restore file system {parent_fs_name} from snapshot {snap_name}: {str(err)}')
    return changed