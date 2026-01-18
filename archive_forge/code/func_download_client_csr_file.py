from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def download_client_csr_file(self, dest_path):
    if not os.path.exists(dest_path):
        try:
            os.makedirs(dest_path)
        except OSError as e:
            self.module.fail_json(msg="Specified destination path '%s' not exist, but failed to create it with exception: %s" % (dest_path, to_native(e)))
    client_csr_file_path = os.path.join(dest_path, self.key_provider_id.id + '_client_csr.pem')
    client_csr = self.crypto_mgr.RetrieveClientCsr(self.key_provider_id)
    if not client_csr:
        try:
            client_csr = self.crypto_mgr.GenerateClientCsr(self.key_provider_id)
        except Exception as e:
            self.module.fail_json(msg='Generate client CSR failed with exception: %s' % to_native(e))
    if not client_csr:
        self.module.fail_json(msg="Generated client CSR is empty '%s'" % client_csr)
    else:
        client_csr_file = open(client_csr_file_path, 'w')
        client_csr_file.write(client_csr)
        client_csr_file.close()
    return client_csr_file_path