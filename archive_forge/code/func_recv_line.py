import errno
import select
import socket
import six
import sys
from ._exceptions import *
from ._ssl_compat import *
from ._utils import *
def recv_line(sock):
    line = []
    while True:
        c = recv(sock, 1)
        line.append(c)
        if c == six.b('\n'):
            break
    return six.b('').join(line)