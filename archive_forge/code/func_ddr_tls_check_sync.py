import socket
import time
from urllib.parse import urlparse
import dns.asyncbackend
import dns.inet
import dns.name
import dns.nameserver
import dns.query
import dns.rdtypes.svcbbase
def ddr_tls_check_sync(self, lifetime):
    ctx = self.make_tls_context()
    expiration = time.time() + lifetime
    with socket.create_connection((self.bootstrap_address, self.port), lifetime) as s:
        with ctx.wrap_socket(s, server_hostname=self.hostname) as ts:
            ts.settimeout(dns.query._remaining(expiration))
            ts.do_handshake()
            cert = ts.getpeercert()
            return self.ddr_check_certificate(cert)