from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def eradicate_snap(module, array):
    """Eradicate a filesystem snapshot"""
    changed = True
    if not module.check_mode:
        snapname = module.params['filesystem'] + ':' + module.params['name'] + '.' + module.params['client'] + '.' + module.params['suffix']
        res = array.delete_directory_snapshots(names=[snapname])
        if res.status_code != 200:
            module.fail_json(msg='Failed to eradicate filesystem snapshot {0}. Error: {1}'.format(snapname, res.errors[0].message))
    module.exit_json(changed=changed)