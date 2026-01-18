from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
def add_openssl_information(module):
    openssl_binary = module.get_bin_path('openssl')
    result = {'openssl_present': openssl_binary is not None}
    if openssl_binary is None:
        return result
    openssl_result = {'path': openssl_binary}
    result['openssl'] = openssl_result
    rc, out, err = module.run_command([openssl_binary, 'version'])
    if rc == 0:
        openssl_result['version_output'] = out
        parts = out.split(None, 2)
        if len(parts) > 1:
            openssl_result['version'] = parts[1]
    return result