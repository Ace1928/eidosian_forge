import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def ssl_set_ca_cert_file(file_name):
    Stream._SSL_ca_cert_file = file_name