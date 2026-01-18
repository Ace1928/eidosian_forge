from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def iso_add_dir(module, opened_iso, iso_type, dir_path):
    parent_dir, check_dirname = dir_path.rsplit('/', 1)
    if not parent_dir.strip():
        parent_dir = '/'
    check_dirname = check_dirname.strip()
    for dirname, dirlist, dummy_filelist in opened_iso.walk(iso_path=parent_dir.upper()):
        if dirname == parent_dir.upper():
            if check_dirname.upper() in dirlist:
                return
            if parent_dir == '/':
                current_dirpath = '/%s' % check_dirname
            else:
                current_dirpath = '%s/%s' % (parent_dir, check_dirname)
            current_dirpath_upper = current_dirpath.upper()
            try:
                if iso_type == 'iso9660':
                    opened_iso.add_directory(current_dirpath_upper)
                elif iso_type == 'rr':
                    opened_iso.add_directory(current_dirpath_upper, rr_name=check_dirname)
                elif iso_type == 'joliet':
                    opened_iso.add_directory(current_dirpath_upper, joliet_path=current_dirpath)
                elif iso_type == 'udf':
                    opened_iso.add_directory(current_dirpath_upper, udf_path=current_dirpath)
            except Exception as err:
                msg = 'Failed to create dir %s with error: %s' % (current_dirpath, to_native(err))
                module.fail_json(msg=msg)