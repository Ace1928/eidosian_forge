import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def ssl_set_certificate_file(file_name):
    Stream._SSL_certificate_file = file_name