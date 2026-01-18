fromfd() -- create a socket object from an open file descriptor [*]
fromshare() -- create a socket object from data received from socket.share() [*]
import _socket
from _socket import *
import os, sys, io, selectors
from enum import IntEnum, IntFlag
def set_inheritable(self, inheritable):
    os.set_inheritable(self.fileno(), inheritable)