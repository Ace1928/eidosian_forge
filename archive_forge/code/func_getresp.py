import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def getresp(self):
    resp = self.getmultiline()
    if self.debugging:
        print('*resp*', self.sanitize(resp))
    self.lastresp = resp[:3]
    c = resp[:1]
    if c in {'1', '2', '3'}:
        return resp
    if c == '4':
        raise error_temp(resp)
    if c == '5':
        raise error_perm(resp)
    raise error_proto(resp)