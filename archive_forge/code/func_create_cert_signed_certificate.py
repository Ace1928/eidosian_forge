import base64
import datetime
from os import remove
from os.path import join
from OpenSSL import crypto
import dateutil.parser
import pytz
import saml2.cryptography.pki
def create_cert_signed_certificate(self, sign_cert_str, sign_key_str, request_cert_str, hash_alg='sha256', valid_from=0, valid_to=315360000, sn=1, passphrase=None):
    """
        Will sign a certificate request with a give certificate.
        :param sign_cert_str:     This certificate will be used to sign with.
                                  Must be a string representation of
                                  the certificate. If you only have a file
                                  use the method read_str_from_file to
                                  get a string representation.
        :param sign_key_str:        This is the key for the ca_cert_str
                                  represented as a string.
                                  If you only have a file use the method
                                  read_str_from_file to get a string
                                  representation.
        :param request_cert_str:  This is the prepared certificate to be
                                  signed. Must be a string representation of
                                  the requested certificate. If you only have
                                  a file use the method read_str_from_file
                                  to get a string representation.
        :param hash_alg:          Hash algorithm to use for the key. Default
                                  is sha256.
        :param valid_from:        When the certificate starts to be valid.
                                  Amount of seconds from when the
                                  certificate is generated.
        :param valid_to:          How long the certificate will be valid from
                                  when it is generated.
                                  The value is in seconds. Default is
                                  315360000 seconds, a.k.a 10 years.
        :param sn:                Serial number for the certificate. Default
                                  is 1.
        :param passphrase:        Password for the private key in sign_key_str.
        :return:                  String representation of the signed
                                  certificate.
        """
    ca_cert = crypto.load_certificate(crypto.FILETYPE_PEM, sign_cert_str)
    ca_key = None
    if passphrase is not None:
        ca_key = crypto.load_privatekey(crypto.FILETYPE_PEM, sign_key_str, passphrase)
    else:
        ca_key = crypto.load_privatekey(crypto.FILETYPE_PEM, sign_key_str)
    req_cert = crypto.load_certificate_request(crypto.FILETYPE_PEM, request_cert_str)
    cert = crypto.X509()
    cert.set_subject(req_cert.get_subject())
    cert.set_serial_number(sn)
    cert.gmtime_adj_notBefore(valid_from)
    cert.gmtime_adj_notAfter(valid_to)
    cert.set_issuer(ca_cert.get_subject())
    cert.set_pubkey(req_cert.get_pubkey())
    cert.sign(ca_key, hash_alg)
    cert_dump = crypto.dump_certificate(crypto.FILETYPE_PEM, cert)
    if isinstance(cert_dump, str):
        return cert_dump
    return cert_dump.decode('utf-8')