import sys
from typing import List, Optional, Union
from twisted.conch.ssh.transport import SSHCiphers, SSHClientTransport
from twisted.python import usage
def opt_ciphers(self, ciphers):
    """Select encryption algorithms"""
    ciphers = ciphers.split(',')
    for cipher in ciphers:
        if cipher not in SSHCiphers.cipherMap:
            sys.exit("Unknown cipher type '%s'" % cipher)
    self['ciphers'] = ciphers