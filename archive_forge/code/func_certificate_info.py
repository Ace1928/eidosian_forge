from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
def certificate_info(self, info, data, path):
    """Load x509 certificate that is either encoded DER or PEM encoding and return the certificate fingerprint."""
    fingerprint = binascii.hexlify(info.fingerprint(info.signature_hash_algorithm)).decode('utf-8')
    return {self.sanitize_distinguished_name(info.subject.rfc4514_string()): {'alias': fingerprint, 'fingerprint': fingerprint, 'certificate': data, 'path': path, 'issuer': self.sanitize_distinguished_name(info.issuer.rfc4514_string())}}