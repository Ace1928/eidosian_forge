from binascii import hexlify
import getpass
import inspect
import os
import socket
import warnings
from errno import ECONNREFUSED, EHOSTUNREACH
from paramiko.agent import Agent
from paramiko.common import DEBUG
from paramiko.config import SSH_PORT
from paramiko.dsskey import DSSKey
from paramiko.ecdsakey import ECDSAKey
from paramiko.ed25519key import Ed25519Key
from paramiko.hostkeys import HostKeys
from paramiko.rsakey import RSAKey
from paramiko.ssh_exception import (
from paramiko.transport import Transport
from paramiko.util import ClosingContextManager
def load_system_host_keys(self, filename=None):
    """
        Load host keys from a system (read-only) file.  Host keys read with
        this method will not be saved back by `save_host_keys`.

        This method can be called multiple times.  Each new set of host keys
        will be merged with the existing set (new replacing old if there are
        conflicts).

        If ``filename`` is left as ``None``, an attempt will be made to read
        keys from the user's local "known hosts" file, as used by OpenSSH,
        and no exception will be raised if the file can't be read.  This is
        probably only useful on posix.

        :param str filename: the filename to read, or ``None``

        :raises: ``IOError`` --
            if a filename was provided and the file could not be read
        """
    if filename is None:
        filename = os.path.expanduser('~/.ssh/known_hosts')
        try:
            self._system_host_keys.load(filename)
        except IOError:
            pass
        return
    self._system_host_keys.load(filename)