from __future__ import absolute_import, division, print_function
import re
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.common import (
def eradicate_volume(module, array):
    """Eradicate Deleted Volume"""
    changed = True
    volfact = []
    if not module.check_mode:
        if module.params['eradicate']:
            try:
                array.eradicate_volume(module.params['name'])
            except Exception:
                module.fail_json(msg='Eradication of volume {0} failed'.format(module.params['name']))
    module.exit_json(changed=changed, volume=volfact)