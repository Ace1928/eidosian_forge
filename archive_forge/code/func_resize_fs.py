from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils._mount import ismount
import re
def resize_fs(module, filesystem, size):
    """ Resize LVM file system. """
    chfs_cmd = module.get_bin_path('chfs', True)
    if not module.check_mode:
        rc, chfs_out, err = module.run_command([chfs_cmd, '-a', 'size=%s' % size, filesystem])
        if rc == 28:
            changed = False
            return (changed, chfs_out)
        elif rc != 0:
            if re.findall('Maximum allocation for logical', err):
                changed = False
                return (changed, err)
            else:
                module.fail_json(msg='Failed to run chfs. Error message: %s' % err)
        else:
            if re.findall('The filesystem size is already', chfs_out):
                changed = False
            else:
                changed = True
            return (changed, chfs_out)
    else:
        changed = True
        msg = ''
        return (changed, msg)