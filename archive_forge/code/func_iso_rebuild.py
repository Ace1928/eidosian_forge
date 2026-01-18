from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def iso_rebuild(module, src_iso, dest_iso, delete_files_list, add_files_list):
    iso = None
    iso_type = 'iso9660'
    try:
        iso = pycdlib.PyCdlib(always_consistent=True)
        iso.open(src_iso)
        if iso.has_rock_ridge():
            iso_type = 'rr'
        elif iso.has_joliet():
            iso_type = 'joliet'
        elif iso.has_udf():
            iso_type = 'udf'
        for item in delete_files_list:
            iso_delete_file(module, iso, iso_type, item)
        for item in add_files_list:
            iso_add_file(module, iso, iso_type, item['src_file'], item['dest_file'])
        iso.write(dest_iso)
    except Exception as err:
        msg = 'Failed to rebuild ISO %s with error: %s' % (src_iso, to_native(err))
        module.fail_json(msg=msg)
    finally:
        if iso:
            iso.close()