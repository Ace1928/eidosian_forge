import socket
import ssl
import struct
import OpenSSL
from glanceclient import exc
def do_verify_callback(connection, x509, errnum, depth, preverify_ok, host=None):
    """Verify the server's SSL certificate.

    This is a standalone function rather than a method to avoid
    issues around closing sockets if a reference is held on
    a VerifiedHTTPSConnection by the callback function.
    """
    if x509.has_expired():
        msg = "SSL Certificate expired on '%s'" % x509.get_notAfter()
        raise exc.SSLCertificateError(msg)
    if depth == 0 and preverify_ok:
        return host_matches_cert(host, x509)
    else:
        return preverify_ok