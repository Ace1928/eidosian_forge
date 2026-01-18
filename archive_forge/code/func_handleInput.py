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
def handleInput(self, char):
    if char in (b'\n', b'\r'):
        self.escapeMode = 1
        self.write(char)
    elif self.escapeMode == 1 and char == options['escape']:
        self.escapeMode = 2
    elif self.escapeMode == 2:
        self.escapeMode = 1
        if char == b'.':
            log.msg('disconnecting from escape')
            stopConnection()
            return
        elif char == b'\x1a':

            def _():
                _leaveRawMode()
                sys.stdout.flush()
                sys.stdin.flush()
                os.kill(os.getpid(), signal.SIGTSTP)
                _enterRawMode()
            reactor.callLater(0, _)
            return
        elif char == b'R':
            log.msg('rekeying connection')
            self.conn.transport.sendKexInit()
            return
        elif char == b'#':
            self.stdio.write(b'\r\nThe following connections are open:\r\n')
            channels = self.conn.channels.keys()
            channels.sort()
            for channelId in channels:
                self.stdio.write(networkString('  #{} {}\r\n'.format(channelId, self.conn.channels[channelId])))
            return
        self.write(b'~' + char)
    else:
        self.escapeMode = 0
        self.write(char)