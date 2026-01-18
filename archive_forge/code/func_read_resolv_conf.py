import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def read_resolv_conf(self, f):
    """Process *f* as a file in the /etc/resolv.conf format.  If f is
        a ``text``, it is used as the name of the file to open; otherwise it
        is treated as the file itself."""
    if isinstance(f, string_types):
        try:
            f = open(f, 'r')
        except IOError:
            self.nameservers = ['127.0.0.1']
            return
        want_close = True
    else:
        want_close = False
    try:
        for l in f:
            if len(l) == 0 or l[0] == '#' or l[0] == ';':
                continue
            tokens = l.split()
            if len(tokens) < 2:
                continue
            if tokens[0] == 'nameserver':
                self.nameservers.append(tokens[1])
            elif tokens[0] == 'domain':
                self.domain = dns.name.from_text(tokens[1])
            elif tokens[0] == 'search':
                for suffix in tokens[1:]:
                    self.search.append(dns.name.from_text(suffix))
            elif tokens[0] == 'options':
                if 'rotate' in tokens[1:]:
                    self.rotate = True
    finally:
        if want_close:
            f.close()
    if len(self.nameservers) == 0:
        self.nameservers.append('127.0.0.1')