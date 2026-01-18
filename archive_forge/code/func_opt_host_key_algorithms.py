import sys
from typing import List, Optional, Union
from twisted.conch.ssh.transport import SSHCiphers, SSHClientTransport
from twisted.python import usage
def opt_host_key_algorithms(self, hkas):
    """Select host key algorithms"""
    if isinstance(hkas, str):
        hkas = hkas.encode('utf-8')
    hkas = hkas.split(b',')
    for hka in hkas:
        if hka not in SSHClientTransport.supportedPublicKeys:
            sys.exit("Unknown host key type '%r'" % hka)
    self['host-key-algorithms'] = hkas