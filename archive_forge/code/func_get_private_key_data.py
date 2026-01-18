from __future__ import absolute_import, division, print_function
import abc
import base64
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.privatekey_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def get_private_key_data(self):
    """Return bytes for self.private_key"""
    try:
        export_format = self._get_wanted_format()
        export_encoding = cryptography.hazmat.primitives.serialization.Encoding.PEM
        if export_format == 'pkcs1':
            export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.TraditionalOpenSSL
        elif export_format == 'pkcs8':
            export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.PKCS8
        elif export_format == 'raw':
            export_format = cryptography.hazmat.primitives.serialization.PrivateFormat.Raw
            export_encoding = cryptography.hazmat.primitives.serialization.Encoding.Raw
    except AttributeError:
        self.module.fail_json(msg='Cryptography backend does not support the selected output format "{0}"'.format(self.format))
    encryption_algorithm = cryptography.hazmat.primitives.serialization.NoEncryption()
    if self.cipher and self.passphrase:
        if self.cipher == 'auto':
            encryption_algorithm = cryptography.hazmat.primitives.serialization.BestAvailableEncryption(to_bytes(self.passphrase))
        else:
            self.module.fail_json(msg='Cryptography backend can only use "auto" for cipher option.')
    try:
        return self.private_key.private_bytes(encoding=export_encoding, format=export_format, encryption_algorithm=encryption_algorithm)
    except ValueError as dummy:
        self.module.fail_json(msg='Cryptography backend cannot serialize the private key in the required format "{0}"'.format(self.format))
    except Exception as dummy:
        self.module.fail_json(msg='Error while serializing the private key in the required format "{0}"'.format(self.format), exception=traceback.format_exc())