import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def handle_accepted(self, conn, addr):
    print('Incoming connection from %s' % repr(addr), file=DEBUGSTREAM)
    channel = self.channel_class(self, conn, addr, self.data_size_limit, self._map, self.enable_SMTPUTF8, self._decode_data)