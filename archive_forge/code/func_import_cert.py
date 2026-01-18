from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.purestorage.flasharray.plugins.module_utils.purefa import (
from ansible_collections.purestorage.flasharray.plugins.module_utils.version import (
def import_cert(module, array, reimport=False):
    """Import a CA provided SSL certificate"""
    changed = True
    certificate = flasharray.CertificatePost(certificate=module.params['certificate'], intermediate_certificate=module.params['intermeadiate_cert'], key=module.params['key'], key_size=module.params['key_size'], passphrase=module.params['passphrase'], status='imported')
    if not module.check_mode:
        if reimport:
            res = array.patch_certificates(names=[module.params['name']], certificate=certificate)
        else:
            res = array.post_certificates(names=[module.params['name']], certificate=certificate)
        if res.status_code != 200:
            module.fail_json(msg='Importing Certificate failed. Error: {0}'.format(res.errors[0].message))
    module.exit_json(changed=changed)