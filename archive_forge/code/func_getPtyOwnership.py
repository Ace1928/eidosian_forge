from __future__ import annotations
import fcntl
import grp
import os
import pty
import pwd
import socket
import struct
import time
import tty
from typing import Callable, Dict, Tuple
from zope.interface import implementer
from twisted.conch import ttymodes
from twisted.conch.avatar import ConchUser
from twisted.conch.error import ConchError
from twisted.conch.interfaces import ISession, ISFTPFile, ISFTPServer
from twisted.conch.ls import lsLine
from twisted.conch.ssh import filetransfer, forwarding, session
from twisted.conch.ssh.filetransfer import (
from twisted.cred import portal
from twisted.cred.error import LoginDenied
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.interfaces import IListeningPort
from twisted.logger import Logger
from twisted.python import components
from twisted.python.compat import nativeString
def getPtyOwnership(self):
    ttyGid = os.stat(self.ptyTuple[2])[5]
    uid, gid = self.avatar.getUserGroupId()
    euid, egid = (os.geteuid(), os.getegid())
    os.setegid(0)
    os.seteuid(0)
    try:
        os.chown(self.ptyTuple[2], uid, ttyGid)
    finally:
        os.setegid(egid)
        os.seteuid(euid)