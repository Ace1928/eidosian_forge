from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def eradicate_endpoint(module, array):
    """Eradicate Deleted Endpoint"""
    changed = True
    volfact = []
    if not module.check_mode:
        if module.params['eradicate']:
            try:
                array.eradicate_volume(module.params['name'], protocol_endpoint=True)
            except Exception:
                module.fail_json(msg='Eradication of endpoint {0} failed'.format(module.params['name']))
    module.exit_json(changed=changed, volume=volfact)