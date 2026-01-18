from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def create_fs_snapshot(module, system):
    """ Create Snapshot from parent fs """
    snapshot_name = module.params['name']
    parent_fs_name = module.params['parent_fs_name']
    changed = False
    if not module.check_mode:
        try:
            parent_fs = system.filesystems.get(name=parent_fs_name)
        except ObjectNotFound:
            msg = f'Cannot create snapshot {snapshot_name}. Parent file system {parent_fs_name} not found'
            module.fail_json(msg=msg)
        if not parent_fs:
            msg = f"Cannot find new snapshot's parent file system named {parent_fs_name}"
            module.fail_json(msg=msg)
        if not module.check_mode:
            if module.params['snapshot_lock_only']:
                msg = "Snapshot does not exist. Cannot comply with 'snapshot_lock_only: true'."
                module.fail_json(msg=msg)
            check_snapshot_lock_options(module)
            snapshot = parent_fs.create_snapshot(name=snapshot_name)
            is_write_prot = snapshot.is_write_protected()
            desired_is_write_prot = module.params['write_protected']
            if is_write_prot != desired_is_write_prot:
                snapshot.update_field('write_protected', desired_is_write_prot)
        manage_snapshot_locks(module, snapshot)
        changed = True
    return changed