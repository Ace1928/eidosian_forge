from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def create_kmip(module, array):
    """Create KMIP object"""
    if array.get_certificates(names=[module.params['certificate']]).status_code != 200:
        module.fail_json(msg='Array certificate {0} does not exist.'.format(module.params['certificate']))
    changed = True
    kmip = flasharray.KmipPost(uris=sorted(module.params['uris']), ca_certificate=module.params['ca_certificate'], certificate=flasharray.ReferenceNoId(name=module.params['certificate']))
    if not module.check_mode:
        res = array.post_kmip(names=[module.params['name']], kmip=kmip)
        if res.status_code != 200:
            module.fail_json(msg='Creating KMIP object {0} failed. Error: {1}'.format(module.params['name'], res.errors[0].message))
    module.exit_json(changed=changed)