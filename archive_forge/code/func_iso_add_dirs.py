from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def iso_add_dirs(module, opened_iso, iso_type, dir_path):
    dirnames = dir_path.strip().split('/')
    current_dirpath = '/'
    for item in dirnames:
        if not item.strip():
            continue
        if current_dirpath == '/':
            current_dirpath = '/%s' % item
        else:
            current_dirpath = '%s/%s' % (current_dirpath, item)
        iso_add_dir(module, opened_iso, iso_type, current_dirpath)