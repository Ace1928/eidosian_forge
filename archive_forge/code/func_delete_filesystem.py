from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@api_wrapper
def delete_filesystem(module, filesystem):
    """ Delete Filesystem """
    changed = False
    if not module.check_mode:
        filesystem.delete()
        changed = True
    return changed