import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def set_dscp(sock, family, dscp):
    if dscp > 63:
        raise ValueError('Invalid dscp %d' % dscp)
    val = dscp << 2
    if family == socket.AF_INET:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, val)
    elif family == socket.AF_INET6:
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_TCLASS, val)
    else:
        raise ValueError('Invalid family %d' % family)