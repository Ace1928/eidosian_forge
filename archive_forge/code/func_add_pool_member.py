from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.proxmox import (proxmox_auth_argument_spec, ProxmoxAnsible)
def add_pool_member(self, poolid, member, member_type):
    current_vms_members, current_storage_members = self.pool_members(poolid)
    all_members_before = current_storage_members + current_vms_members
    all_members_after = all_members_before.copy()
    diff = {'before': {'members': all_members_before}, 'after': {'members': all_members_after}}
    try:
        if member_type == 'storage':
            storages = self.get_storages(type=None)
            if member not in [storage['storage'] for storage in storages]:
                self.module.fail_json(msg="Storage {0} doesn't exist in the cluster".format(member))
            if member in current_storage_members:
                self.module.exit_json(changed=False, poolid=poolid, member=member, diff=diff, msg='Member {0} is already part of the pool {1}'.format(member, poolid))
            all_members_after.append(member)
            if self.module.check_mode:
                return diff
            self.proxmox_api.pools(poolid).put(storage=[member])
            return diff
        else:
            try:
                vmid = int(member)
            except ValueError:
                vmid = self.get_vmid(member)
            if vmid in current_vms_members:
                self.module.exit_json(changed=False, poolid=poolid, member=member, diff=diff, msg='VM {0} is already part of the pool {1}'.format(member, poolid))
            all_members_after.append(member)
            if not self.module.check_mode:
                self.proxmox_api.pools(poolid).put(vms=[vmid])
            return diff
    except Exception as e:
        self.module.fail_json(msg='Failed to add a new member ({0}) to the pool {1}: {2}'.format(member, poolid, e))