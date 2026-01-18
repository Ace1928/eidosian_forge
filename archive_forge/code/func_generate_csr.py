from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def generate_csr(self):
    """(Re-)Generate CSR."""
    self._ensure_private_key_loaded()
    csr = cryptography.x509.CertificateSigningRequestBuilder()
    try:
        csr = csr.subject_name(cryptography.x509.Name([cryptography.x509.NameAttribute(cryptography_name_to_oid(entry[0]), to_text(entry[1])) for entry in self.subject]))
    except ValueError as e:
        raise CertificateSigningRequestError(e)
    if self.subjectAltName:
        csr = csr.add_extension(cryptography.x509.SubjectAlternativeName([cryptography_get_name(name) for name in self.subjectAltName]), critical=self.subjectAltName_critical)
    if self.keyUsage:
        params = cryptography_parse_key_usage_params(self.keyUsage)
        csr = csr.add_extension(cryptography.x509.KeyUsage(**params), critical=self.keyUsage_critical)
    if self.extendedKeyUsage:
        usages = [cryptography_name_to_oid(usage) for usage in self.extendedKeyUsage]
        csr = csr.add_extension(cryptography.x509.ExtendedKeyUsage(usages), critical=self.extendedKeyUsage_critical)
    if self.basicConstraints:
        params = {}
        ca, path_length = cryptography_get_basic_constraints(self.basicConstraints)
        csr = csr.add_extension(cryptography.x509.BasicConstraints(ca, path_length), critical=self.basicConstraints_critical)
    if self.ocspMustStaple:
        try:
            csr = csr.add_extension(cryptography.x509.TLSFeature([cryptography.x509.TLSFeatureType.status_request]), critical=self.ocspMustStaple_critical)
        except AttributeError as dummy:
            csr = csr.add_extension(cryptography.x509.UnrecognizedExtension(CRYPTOGRAPHY_MUST_STAPLE_NAME, CRYPTOGRAPHY_MUST_STAPLE_VALUE), critical=self.ocspMustStaple_critical)
    if self.name_constraints_permitted or self.name_constraints_excluded:
        try:
            csr = csr.add_extension(cryptography.x509.NameConstraints([cryptography_get_name(name, 'name constraints permitted') for name in self.name_constraints_permitted] or None, [cryptography_get_name(name, 'name constraints excluded') for name in self.name_constraints_excluded] or None), critical=self.name_constraints_critical)
        except TypeError as e:
            raise OpenSSLObjectError('Error while parsing name constraint: {0}'.format(e))
    if self.create_subject_key_identifier:
        csr = csr.add_extension(cryptography.x509.SubjectKeyIdentifier.from_public_key(self.privatekey.public_key()), critical=False)
    elif self.subject_key_identifier is not None:
        csr = csr.add_extension(cryptography.x509.SubjectKeyIdentifier(self.subject_key_identifier), critical=False)
    if self.authority_key_identifier is not None or self.authority_cert_issuer is not None or self.authority_cert_serial_number is not None:
        issuers = None
        if self.authority_cert_issuer is not None:
            issuers = [cryptography_get_name(n, 'authority cert issuer') for n in self.authority_cert_issuer]
        csr = csr.add_extension(cryptography.x509.AuthorityKeyIdentifier(self.authority_key_identifier, issuers, self.authority_cert_serial_number), critical=False)
    if self.crl_distribution_points:
        csr = csr.add_extension(cryptography.x509.CRLDistributionPoints(self.crl_distribution_points), critical=False)
    digest = None
    if cryptography_key_needs_digest_for_signing(self.privatekey):
        digest = select_message_digest(self.digest)
        if digest is None:
            raise CertificateSigningRequestError('Unsupported digest "{0}"'.format(self.digest))
    try:
        self.csr = csr.sign(self.privatekey, digest, self.cryptography_backend)
    except TypeError as e:
        if str(e) == 'Algorithm must be a registered hash algorithm.' and digest is None:
            self.module.fail_json(msg='Signing with Ed25519 and Ed448 keys requires cryptography 2.8 or newer.')
        raise
    except UnicodeError as e:
        msg = 'Error while creating CSR: {0}\n'.format(e)
        if self.using_common_name_for_san:
            self.module.fail_json(msg=msg + 'This is probably caused because the Common Name is used as a SAN. Specifying use_common_name_for_san=false might fix this.')
        self.module.fail_json(msg=msg + 'This is probably caused by an invalid Subject Alternative DNS Name.')