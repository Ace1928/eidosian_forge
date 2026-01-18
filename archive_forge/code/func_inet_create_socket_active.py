import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def inet_create_socket_active(style, address):
    try:
        is_addr_inet = is_valid_ipv4_address(address[0])
        if is_addr_inet:
            sock = socket.socket(socket.AF_INET, style, 0)
            family = socket.AF_INET
        else:
            sock = socket.socket(socket.AF_INET6, style, 0)
            family = socket.AF_INET6
    except socket.error as e:
        return (get_exception_errno(e), None)
    return (family, sock)