import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def inet_open_active(style, target, default_port, dscp):
    address = inet_parse_active(target, default_port)
    family, sock = inet_create_socket_active(style, address)
    if sock is None:
        return (family, sock)
    error = inet_connect_active(sock, address, family, dscp)
    if error:
        return (error, None)
    return (0, sock)