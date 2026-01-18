import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def check_connection_completion(sock):
    try:
        return Stream.check_connection_completion(sock)
    except ssl.SSLSyscallError as e:
        return ovs.socket_util.get_exception_errno(e)