import socket
import ssl
import struct
import OpenSSL
from glanceclient import exc
def host_matches_cert(host, x509):
    """Verify the certificate identifies the host.

    Verify that the x509 certificate we have received
    from 'host' correctly identifies the server we are
    connecting to, ie that the certificate's Common Name
    or a Subject Alternative Name matches 'host'.
    """

    def check_match(name):
        if name == host:
            return True
        if name.startswith('*.') and host.find('.') > 0:
            if name[2:] == host.split('.', 1)[1]:
                return True
    common_name = x509.get_subject().commonName
    if check_match(common_name):
        return True
    san_list = None
    for i in range(x509.get_extension_count()):
        ext = x509.get_extension(i)
        if ext.get_short_name() == b'subjectAltName':
            san_list = str(ext)
            for san in ''.join(san_list.split()).split(','):
                if san.startswith('DNS:'):
                    if check_match(san.split(':', 1)[1]):
                        return True
    msg = 'Host "%s" does not match x509 certificate contents: CommonName "%s"' % (host, common_name)
    if san_list is not None:
        msg = msg + ', subjectAltName "%s"' % san_list
    raise exc.SSLCertificateError(msg)