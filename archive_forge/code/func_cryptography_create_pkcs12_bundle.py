from __future__ import (absolute_import, division, print_function)
import os
import re
import tempfile
from ansible.module_utils.six import PY2
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
def cryptography_create_pkcs12_bundle(self, keystore_p12_path, key_format='PEM', cert_format='PEM'):
    if key_format == 'PEM':
        key_loader = load_pem_private_key
    else:
        key_loader = load_der_private_key
    if cert_format == 'PEM':
        cert_loader = load_pem_x509_certificate
    else:
        cert_loader = load_der_x509_certificate
    try:
        with open(self.private_key_path, 'rb') as key_file:
            private_key = key_loader(key_file.read(), password=to_bytes(self.keypass), backend=backend)
    except TypeError:
        try:
            with open(self.private_key_path, 'rb') as key_file:
                private_key = key_loader(key_file.read(), password=None, backend=backend)
        except (OSError, TypeError, ValueError, UnsupportedAlgorithm) as e:
            self.module.fail_json(msg='The following error occurred while loading the provided private_key: %s' % to_native(e))
    except (OSError, ValueError, UnsupportedAlgorithm) as e:
        self.module.fail_json(msg='The following error occurred while loading the provided private_key: %s' % to_native(e))
    try:
        with open(self.certificate_path, 'rb') as cert_file:
            cert = cert_loader(cert_file.read(), backend=backend)
    except (OSError, ValueError, UnsupportedAlgorithm) as e:
        self.module.fail_json(msg='The following error occurred while loading the provided certificate: %s' % to_native(e))
    if self.password:
        encryption = BestAvailableEncryption(to_bytes(self.password))
    else:
        encryption = NoEncryption()
    pkcs12_bundle = serialize_key_and_certificates(name=to_bytes(self.name), key=private_key, cert=cert, cas=None, encryption_algorithm=encryption)
    with open(keystore_p12_path, 'wb') as p12_file:
        p12_file.write(pkcs12_bundle)
    self.result.update(msg='PKCS#12 bundle created by cryptography backend')