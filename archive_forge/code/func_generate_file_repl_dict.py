from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flashblade.plugins.module_utils.purefb import (
from datetime import datetime
def generate_file_repl_dict(blade):
    file_repl_info = {}
    file_links = blade.file_system_replica_links.list_file_system_replica_links()
    for linkcnt in range(0, len(file_links.items)):
        fs_name = file_links.items[linkcnt].local_file_system.name
        file_repl_info[fs_name] = {'direction': file_links.items[linkcnt].direction, 'lag': file_links.items[linkcnt].lag, 'status': file_links.items[linkcnt].status, 'remote_fs': file_links.items[linkcnt].remote.name + ':' + file_links.items[linkcnt].remote_file_system.name, 'recovery_point': file_links.items[linkcnt].recovery_point}
        file_repl_info[fs_name]['policies'] = []
        for policy_cnt in range(0, len(file_links.items[linkcnt].policies)):
            file_repl_info[fs_name]['policies'].append(file_links.items[linkcnt].policies[policy_cnt].display_name)
    return file_repl_info