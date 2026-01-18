import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def get_peer_creds(self):
    """Return the PID/UID/GID tuple of the peer socket for UNIX sockets.

        This function uses SO_PEERCRED to query the UNIX PID, UID, GID
        of the peer, which is only available if the bind address is
        a UNIX domain socket.

        Raises:
            NotImplementedError: in case of unsupported socket type
            RuntimeError: in case of SO_PEERCRED lookup unsupported or disabled

        """
    PEERCRED_STRUCT_DEF = '3i'
    if IS_WINDOWS or self.socket.family != socket.AF_UNIX:
        raise NotImplementedError('SO_PEERCRED is only supported in Linux kernel and WSL')
    elif not self.peercreds_enabled:
        raise RuntimeError('Peer creds lookup is disabled within this server')
    try:
        peer_creds = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize(PEERCRED_STRUCT_DEF))
    except socket.error as socket_err:
        "Non-Linux kernels don't support SO_PEERCRED.\n\n            Refs:\n            http://welz.org.za/notes/on-peer-cred.html\n            https://github.com/daveti/tcpSockHack\n            msdn.microsoft.com/en-us/commandline/wsl/release_notes#build-15025\n            "
        raise RuntimeError from socket_err
    else:
        pid, uid, gid = struct.unpack(PEERCRED_STRUCT_DEF, peer_creds)
        return (pid, uid, gid)