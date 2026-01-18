from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
from datetime import datetime
import time
def generate_filesystems_dict(array):
    files_info = {}
    filesystems = list(array.get_file_systems().items)
    for filesystem in range(0, len(filesystems)):
        fs_name = filesystems[filesystem].name
        files_info[fs_name] = {'destroyed': filesystems[filesystem].destroyed, 'directories': {}}
        directories = list(array.get_directories(file_system_names=[fs_name]).items)
        for directory in range(0, len(directories)):
            d_name = directories[directory].directory_name
            files_info[fs_name]['directories'][d_name] = {'path': directories[directory].path, 'data_reduction': directories[directory].space.data_reduction, 'snapshots_space': directories[directory].space.snapshots, 'total_physical_space': directories[directory].space.total_physical, 'unique_space': directories[directory].space.unique, 'virtual_space': directories[directory].space.virtual, 'destroyed': directories[directory].destroyed, 'full_name': directories[directory].name, 'used_provisioned': getattr(directories[directory].space, 'used_provisioned', None), 'exports': {}}
            if LooseVersion(SUBS_API_VERSION) <= LooseVersion(array.get_rest_version()):
                files_info[fs_name]['directories'][d_name]['total_used'] = directories[directory].space.total_used
            exports = list(array.get_directory_exports(directory_names=[files_info[fs_name]['directories'][d_name]['full_name']]).items)
            for export in range(0, len(exports)):
                e_name = exports[export].export_name
                files_info[fs_name]['directories'][d_name]['exports'][e_name] = {'enabled': exports[export].enabled, 'policy': {'name': exports[export].policy.name, 'type': exports[export].policy.resource_type}}
    return files_info