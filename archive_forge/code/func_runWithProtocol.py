import os
import sys
import termios
import tty
from twisted.conch.insults.insults import ServerProtocol
from twisted.conch.manhole import ColoredManhole
from twisted.internet import defer, protocol, reactor, stdio
from twisted.python import failure, log, reflect
def runWithProtocol(klass):
    fd = sys.__stdin__.fileno()
    oldSettings = termios.tcgetattr(fd)
    tty.setraw(fd)
    try:
        stdio.StandardIO(ServerProtocol(klass))
        reactor.run()
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, oldSettings)
        os.write(fd, b'\r\x1bc\r')