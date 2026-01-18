from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import re
import tempfile
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
def get_csr_identifiers(self, csr_filename=None, csr_content=None):
    """
        Return a set of requested identifiers (CN and SANs) for the CSR.
        Each identifier is a pair (type, identifier), where type is either
        'dns' or 'ip'.
        """
    filename = csr_filename
    data = None
    if csr_content is not None:
        filename = '/dev/stdin'
        data = csr_content.encode('utf-8')
    openssl_csr_cmd = [self.openssl_binary, 'req', '-in', filename, '-noout', '-text']
    dummy, out, dummy = self.module.run_command(openssl_csr_cmd, data=data, check_rc=True, binary_data=True, environ_update=_OPENSSL_ENVIRONMENT_UPDATE)
    identifiers = set([])
    common_name = re.search('Subject:.* CN\\s?=\\s?([^\\s,;/]+)', to_text(out, errors='surrogate_or_strict'))
    if common_name is not None:
        identifiers.add(('dns', common_name.group(1)))
    subject_alt_names = re.search('X509v3 Subject Alternative Name: (?:critical)?\\n +([^\\n]+)\\n', to_text(out, errors='surrogate_or_strict'), re.MULTILINE | re.DOTALL)
    if subject_alt_names is not None:
        for san in subject_alt_names.group(1).split(', '):
            if san.lower().startswith('dns:'):
                identifiers.add(('dns', san[4:]))
            elif san.lower().startswith('ip:'):
                identifiers.add(('ip', self._normalize_ip(san[3:])))
            elif san.lower().startswith('ip address:'):
                identifiers.add(('ip', self._normalize_ip(san[11:])))
            else:
                raise BackendException('Found unsupported SAN identifier "{0}"'.format(san))
    return identifiers