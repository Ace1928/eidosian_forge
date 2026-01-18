from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def iso_delete_file(module, opened_iso, iso_type, dest_file):
    dest_file = dest_file.strip()
    if dest_file[0] != '/':
        dest_file = '/%s' % dest_file
    file_name = os.path.basename(dest_file)
    if not iso_check_file_exists(opened_iso, dest_file):
        module.fail_json(msg='The file %s does not exist.' % dest_file)
    if '.' not in file_name:
        file_in_iso_path = dest_file.upper() + '.;1'
    else:
        file_in_iso_path = dest_file.upper() + ';1'
    try:
        if iso_type == 'iso9660':
            opened_iso.rm_file(iso_path=file_in_iso_path)
        elif iso_type == 'rr':
            opened_iso.rm_file(iso_path=file_in_iso_path)
        elif iso_type == 'joliet':
            opened_iso.rm_file(joliet_path=dest_file)
        elif iso_type == 'udf':
            opened_iso.rm_file(udf_path=dest_file)
    except Exception as err:
        msg = 'Failed to delete iso file %s with error: %s' % (dest_file, to_native(err))
        module.fail_json(msg=msg)