from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_fs_dict(module, blade):
    api_version = blade.api_version.list_versions().versions
    if SMB_MODE_API_VERSION in api_version:
        bladev2 = get_system(module)
        fsys_v2 = list(bladev2.get_file_systems().items)
    fs_info = {}
    fsys = blade.file_systems.list_file_systems()
    for fsystem in range(0, len(fsys.items)):
        share = fsys.items[fsystem].name
        fs_info[share] = {'fast_remove': fsys.items[fsystem].fast_remove_directory_enabled, 'snapshot_enabled': fsys.items[fsystem].snapshot_directory_enabled, 'provisioned': fsys.items[fsystem].provisioned, 'destroyed': fsys.items[fsystem].destroyed, 'nfs_rules': fsys.items[fsystem].nfs.rules, 'nfs_v3': getattr(fsys.items[fsystem].nfs, 'v3_enabled', False), 'nfs_v4_1': getattr(fsys.items[fsystem].nfs, 'v4_1_enabled', False), 'user_quotas': {}, 'group_quotas': {}}
        if fsys.items[fsystem].http.enabled:
            fs_info[share]['http'] = fsys.items[fsystem].http.enabled
        if fsys.items[fsystem].smb.enabled:
            fs_info[share]['smb_mode'] = fsys.items[fsystem].smb.acl_mode
        api_version = blade.api_version.list_versions().versions
        if MULTIPROTOCOL_API_VERSION in api_version:
            fs_info[share]['multi_protocol'] = {'safegaurd_acls': fsys.items[fsystem].multi_protocol.safeguard_acls, 'access_control_style': fsys.items[fsystem].multi_protocol.access_control_style}
        if HARD_LIMIT_API_VERSION in api_version:
            fs_info[share]['hard_limit'] = fsys.items[fsystem].hard_limit_enabled
        if REPLICATION_API_VERSION in api_version:
            fs_info[share]['promotion_status'] = fsys.items[fsystem].promotion_status
            fs_info[share]['requested_promotion_state'] = fsys.items[fsystem].requested_promotion_state
            fs_info[share]['writable'] = fsys.items[fsystem].writable
            fs_info[share]['source'] = {'is_local': fsys.items[fsystem].source.is_local, 'name': fsys.items[fsystem].source.name}
        if SMB_MODE_API_VERSION in api_version:
            for v2fs in range(0, len(fsys_v2)):
                if fsys_v2[v2fs].name == share:
                    fs_info[share]['default_group_quota'] = fsys_v2[v2fs].default_group_quota
                    fs_info[share]['default_user_quota'] = fsys_v2[v2fs].default_user_quota
                    if NFS_POLICY_API_VERSION in api_version:
                        fs_info[share]['export_policy'] = fsys_v2[v2fs].nfs.export_policy.name
        if VSO_VERSION in api_version:
            for v2fs in range(0, len(fsys_v2)):
                if fsys_v2[v2fs].name == share:
                    try:
                        fs_groups = True
                        fs_group_quotas = list(bladev2.get_quotas_groups(file_system_names=[share]).items)
                    except Exception:
                        fs_groups = False
                    try:
                        fs_users = True
                        fs_user_quotas = list(bladev2.get_quotas_users(file_system_names=[share]).items)
                    except Exception:
                        fs_users = False
                    if fs_groups:
                        for group_quota in range(0, len(fs_group_quotas)):
                            group_name = fs_group_quotas[group_quota].name.rsplit('/')[1]
                            fs_info[share]['group_quotas'][group_name] = {'group_id': fs_group_quotas[group_quota].group.id, 'group_name': fs_group_quotas[group_quota].group.name, 'quota': fs_group_quotas[group_quota].quota, 'usage': fs_group_quotas[group_quota].usage}
                    if fs_users:
                        for user_quota in range(0, len(fs_user_quotas)):
                            user_name = fs_user_quotas[user_quota].name.rsplit('/')[1]
                            fs_info[share]['user_quotas'][user_name] = {'user_id': fs_user_quotas[user_quota].user.id, 'user_name': fs_user_quotas[user_quota].user.name, 'quota': fs_user_quotas[user_quota].quota, 'usage': fs_user_quotas[user_quota].usage}
            if PUBLIC_API_VERSION in api_version:
                for v2fs in range(0, len(fsys_v2)):
                    if fsys_v2[v2fs].name == share:
                        fs_info[share]['smb_client_policy'] = getattr(fsys_v2[v2fs].smb.client_policy, 'name', None)
                        fs_info[share]['smb_share_policy'] = getattr(fsys_v2[v2fs].smb.share_policy, 'name', None)
                        fs_info[share]['smb_continuous_availability_enabled'] = fsys_v2[v2fs].smb.continuous_availability_enabled
                        fs_info[share]['multi_protocol_access_control_style'] = getattr(fsys_v2[v2fs].multi_protocol, 'access_control_style', None)
                        fs_info[share]['multi_protocol_safeguard_acls'] = fsys_v2[v2fs].multi_protocol.safeguard_acls
    return fs_info