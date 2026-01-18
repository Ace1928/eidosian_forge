from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi
def download_upload_cert_for_trust(self, kms_trust_vc_config):
    changed = False
    cert_info = ''
    client_cert = kms_trust_vc_config.get('upload_client_cert')
    client_key = kms_trust_vc_config.get('upload_client_key')
    kms_signed_csr = kms_trust_vc_config.get('upload_kms_signed_client_csr')
    self_signed_cert_path = kms_trust_vc_config.get('download_self_signed_cert')
    client_csr_path = kms_trust_vc_config.get('download_client_csr')
    if client_cert and client_key:
        if not os.path.exists(client_cert) or not os.path.exists(client_key):
            self.module.fail_json(msg="Configured 'upload_client_cert' file: '%s', or 'upload_client_key' file: '%s' does not exist." % (client_cert, client_key))
        self.upload_client_cert_key(client_cert, client_key)
        cert_info = "Client cert file '%s', key file '%s' uploaded for key provider '%s'" % (client_cert, client_key, self.key_provider_id.id)
        changed = True
    elif kms_signed_csr:
        if not os.path.exists(kms_signed_csr):
            self.module.fail_json(msg="Configured 'upload_kms_signed_client_csr' file: '%s' does not exist." % kms_signed_csr)
        self.upload_kms_signed_csr(kms_signed_csr)
        cert_info = "KMS signed client CSR '%s' uploaded for key provider '%s'" % (kms_signed_csr, self.key_provider_id.id)
        changed = True
    elif self_signed_cert_path:
        cert_file_path = self.update_self_signed_client_cert(self_signed_cert_path)
        cert_info = "Client self signed certificate file '%s' for key provider '%s' updated and downloaded" % (cert_file_path, self.key_provider_id.id)
        changed = True
    elif client_csr_path:
        cert_file_path = self.download_client_csr_file(client_csr_path)
        cert_info = "Client certificate signing request file '%s' for key provider '%s' downloaded" % (cert_file_path, self.key_provider_id.id)
    return (changed, cert_info)