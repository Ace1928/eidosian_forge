import socket
import time
from urllib.parse import urlparse
import dns.asyncbackend
import dns.inet
import dns.name
import dns.nameserver
import dns.query
import dns.rdtypes.svcbbase
def ddr_check_certificate(self, cert):
    """Verify that the _SVCBInfo's address is in the cert's subjectAltName (SAN)"""
    for name, value in cert['subjectAltName']:
        if name == 'IP Address' and value == self.bootstrap_address:
            return True
    return False