import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
def connect_wait(self, poller):
    self.wait(poller, Stream.W_CONNECT)