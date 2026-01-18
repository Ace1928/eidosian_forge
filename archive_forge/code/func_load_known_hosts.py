import base64
import getpass
import os
import re
import six
import sys
import socket
import threading
from binascii import hexlify
from ncclient.capabilities import Capabilities
from ncclient.logging_ import SessionLoggerAdapter
import paramiko
from ncclient.transport.errors import AuthenticationError, SSHError, SSHUnknownHostError
from ncclient.transport.session import Session
from ncclient.transport.parser import DefaultXMLParser
import logging
def load_known_hosts(self, filename=None):
    """Load host keys from an openssh :file:`known_hosts`-style file. Can
        be called multiple times.

        If *filename* is not specified, looks in the default locations i.e. :file:`~/.ssh/known_hosts` and :file:`~/ssh/known_hosts` for Windows.
        """
    if filename is None:
        filename = os.path.expanduser('~/.ssh/known_hosts')
        try:
            self._host_keys.load(filename)
        except IOError:
            filename = os.path.expanduser('~/ssh/known_hosts')
            try:
                self._host_keys.load(filename)
            except IOError:
                pass
    else:
        self._host_keys.load(filename)