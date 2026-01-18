from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
def delete_multi_dns(module, array):
    """Delete a DNS configuration"""
    changed = True
    if module.params['name'] == 'management':
        res = array.patch_dns(names=[module.params['name']], dns=flasharray.DnsPatch(domain='', nameservers=[]))
        if res.status_code != 200:
            module.fail_json(msg='Management DNS configuration not deleted. Error: {0}'.format(res.errors[0].message))
    elif not module.check_mode:
        res = array.delete_dns(names=[module.params['name']])
        if res.status_code != 200:
            module.fail_json(msg='Failed to delete DNS configuration {0}. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)