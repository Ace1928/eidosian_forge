import re
import struct
import sys
import eventlet
from eventlet import patcher
from eventlet.green import _socket_nodns
from eventlet.green import os
from eventlet.green import time
from eventlet.green import select
from eventlet.green import ssl
def getaliases(self, hostname):
    """Return a list of all the aliases of a given hostname"""
    if self._hosts:
        aliases = self._hosts.getaliases(hostname)
    else:
        aliases = []
    while True:
        try:
            ans = self._resolver.query(hostname, dns.rdatatype.CNAME)
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            break
        else:
            aliases.extend((str(rr.target) for rr in ans.rrset))
            hostname = ans[0].target
    return aliases