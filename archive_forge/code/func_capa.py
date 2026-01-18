import errno
import re
import socket
import sys
def capa(self):
    """Return server capabilities (RFC 2449) as a dictionary
        >>> c=poplib.POP3('localhost')
        >>> c.capa()
        {'IMPLEMENTATION': ['Cyrus', 'POP3', 'server', 'v2.2.12'],
         'TOP': [], 'LOGIN-DELAY': ['0'], 'AUTH-RESP-CODE': [],
         'EXPIRE': ['NEVER'], 'USER': [], 'STLS': [], 'PIPELINING': [],
         'UIDL': [], 'RESP-CODES': []}
        >>>

        Really, according to RFC 2449, the cyrus folks should avoid
        having the implementation split into multiple arguments...
        """

    def _parsecap(line):
        lst = line.decode('ascii').split()
        return (lst[0], lst[1:])
    caps = {}
    try:
        resp = self._longcmd('CAPA')
        rawcaps = resp[1]
        for capline in rawcaps:
            capnm, capargs = _parsecap(capline)
            caps[capnm] = capargs
    except error_proto:
        raise error_proto('-ERR CAPA not supported by server')
    return caps