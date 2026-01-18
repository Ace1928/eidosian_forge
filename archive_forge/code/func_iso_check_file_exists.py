from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def iso_check_file_exists(opened_iso, dest_file):
    file_dir = os.path.dirname(dest_file).strip()
    file_name = os.path.basename(dest_file)
    dirnames = file_dir.strip().split('/')
    parent_dir = '/'
    for item in dirnames:
        if not item.strip():
            continue
        for dirname, dirlist, dummy_filelist in opened_iso.walk(iso_path=parent_dir.upper()):
            if dirname != parent_dir.upper():
                break
            if item.upper() not in dirlist:
                return False
        if parent_dir == '/':
            parent_dir = '/%s' % item
        else:
            parent_dir = '%s/%s' % (parent_dir, item)
    if '.' not in file_name:
        file_in_iso_path = file_name.upper() + '.;1'
    else:
        file_in_iso_path = file_name.upper() + ';1'
    for dirname, dummy_dirlist, filelist in opened_iso.walk(iso_path=parent_dir.upper()):
        if dirname != parent_dir.upper():
            return False
        return file_name.upper() in filelist or file_in_iso_path in filelist