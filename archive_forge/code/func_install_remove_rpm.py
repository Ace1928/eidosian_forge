from __future__ import absolute_import, division, print_function
import time
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
def install_remove_rpm(module, full_pkg, file_system, state):
    commands = []
    reload_patch = False
    splitted_pkg = full_pkg.split('.')
    pkg = '.'.join(splitted_pkg[0:-1])
    show_inactive = 'show install inactive'
    show_active = 'show install active'
    show_commit = 'show install committed'
    show_patches = 'show install patches'
    show_pkg_info = 'show install pkg-info {0}'.format(pkg)
    if state == 'present':
        inactive_body = execute_show_command(show_inactive, module)
        active_body = execute_show_command(show_active, module)
        if pkg not in inactive_body and pkg not in active_body:
            commands.append(add_operation(module, show_inactive, file_system, full_pkg, pkg))
        patch_type_body = execute_show_command(show_pkg_info, module)
        if patch_type_body and 'Patch Type    :  reload' in patch_type_body:
            reload_patch = True
        if pkg not in active_body:
            if reload_patch:
                commands.append(activate_reload(module, pkg, True))
                return commands
            else:
                commands.append(activate_operation(module, show_active, pkg))
        commit_body = execute_show_command(show_commit, module)
        if pkg not in commit_body:
            patch_body = execute_show_command(show_patches, module)
            if pkg in patch_body:
                commands.append(commit_operation(module, show_commit, pkg, False))
            else:
                err = 'Operation "install activate {0} forced" Failed'.format(pkg)
                module.fail_json(msg=err)
    else:
        commit_body = execute_show_command(show_commit, module)
        active_body = execute_show_command(show_active, module)
        patch_type_body = execute_show_command(show_pkg_info, module)
        if patch_type_body and 'Patch Type    :  reload' in patch_type_body:
            reload_patch = True
        if pkg in commit_body and pkg in active_body:
            if reload_patch:
                commands.append(activate_reload(module, pkg, False))
                return commands
            else:
                commands.append(deactivate_operation(module, show_active, pkg, True))
                commit_body = execute_show_command(show_commit, module)
                if pkg in commit_body:
                    commands.append(commit_operation(module, show_commit, pkg, True))
                commands.extend(remove_operation(module, show_inactive, pkg))
        elif pkg in commit_body:
            commands.append(commit_operation(module, show_commit, pkg, True))
            commands.extend(remove_operation(module, show_inactive, pkg))
        elif pkg in active_body:
            if reload_patch:
                commands.append(activate_reload(module, pkg, False))
                return commands
            else:
                commands.append(deactivate_operation(module, show_inactive, pkg, False))
                commands.extend(remove_operation(module, show_inactive, pkg))
        else:
            inactive_body = execute_show_command(show_inactive, module)
            if pkg in inactive_body:
                commands.extend(remove_operation(module, show_inactive, pkg))
    return commands