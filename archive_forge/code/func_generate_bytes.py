from __future__ import absolute_import, division, print_function
import abc
import base64
import os
import stat
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.io import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
def generate_bytes(self, module):
    """Generate PKCS#12 file archive."""
    pkey = None
    if self.privatekey_content:
        try:
            pkey = load_privatekey(None, content=self.privatekey_content, passphrase=self.privatekey_passphrase, backend=self.backend)
        except OpenSSLBadPassphraseError as exc:
            raise PkcsError(exc)
    cert = None
    if self.certificate_path:
        cert = load_certificate(self.certificate_path, backend=self.backend)
    friendly_name = to_bytes(self.friendly_name) if self.friendly_name is not None else None
    self.pkcs12 = (pkey, cert, self.other_certificates, friendly_name)
    if not self.passphrase:
        encryption = serialization.NoEncryption()
    elif self.encryption_level == 'compatibility2022':
        encryption = serialization.PrivateFormat.PKCS12.encryption_builder().kdf_rounds(self.iter_size).key_cert_algorithm(PBES.PBESv1SHA1And3KeyTripleDESCBC).hmac_hash(hashes.SHA1()).build(to_bytes(self.passphrase))
    else:
        encryption = serialization.BestAvailableEncryption(to_bytes(self.passphrase))
    return serialize_key_and_certificates(friendly_name, pkey, cert, self.other_certificates, encryption)