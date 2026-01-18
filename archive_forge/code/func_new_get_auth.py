import fcntl
import os
import platform
import re
import socket
from Xlib import error, xauth
def new_get_auth(sock, dname, host, dno):
    if uname[0] == 'Darwin' and host and host.startswith('/tmp/'):
        family = xauth.FamilyLocal
        addr = socket.gethostname()
    elif host:
        family = xauth.FamilyInternet
        octets = sock.getpeername()[0].split('.')
        addr = ''.join(map(lambda x: chr(int(x)), octets))
    else:
        family = xauth.FamilyLocal
        addr = socket.gethostname()
    au = xauth.Xauthority()
    while 1:
        try:
            return au.get_best_auth(family, addr, dno)
        except error.XNoAuthError:
            pass
        if family == xauth.FamilyInternet and addr == '\x7f\x00\x00\x01':
            family = xauth.FamilyLocal
            addr = socket.gethostname()
        else:
            return ('', '')