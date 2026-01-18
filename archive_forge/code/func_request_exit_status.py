import fcntl
import getpass
import os
import signal
import struct
import sys
import tty
from typing import List, Tuple
from twisted.conch.client import connect, default
from twisted.conch.client.options import ConchOptions
from twisted.conch.error import ConchError
from twisted.conch.ssh import channel, common, connection, forwarding, session
from twisted.internet import reactor, stdio, task
from twisted.python import log, usage
from twisted.python.compat import ioType, networkString
def request_exit_status(self, data):
    global exitStatus
    exitStatus = int(struct.unpack('>L', data)[0])
    log.msg(f'exit status: {exitStatus}')