import os
import socket
import struct
import sys
import threading
import time
import tempfile
import stat
from logging import DEBUG
from select import select
from paramiko.common import io_sleep, byte_chr
from paramiko.ssh_exception import SSHException, AuthenticationException
from paramiko.message import Message
from paramiko.pkey import PKey, UnknownKeyType
from paramiko.util import asbytes, get_logger
def get_agent_connection():
    """
    Returns some SSH agent object, or None if none were found/supported.

    .. versionadded:: 2.10
    """
    if 'SSH_AUTH_SOCK' in os.environ and sys.platform != 'win32':
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.connect(os.environ['SSH_AUTH_SOCK'])
            return conn
        except:
            return
    elif sys.platform == 'win32':
        from . import win_pageant, win_openssh
        conn = None
        if win_pageant.can_talk_to_agent():
            conn = win_pageant.PageantConnection()
        elif win_openssh.can_talk_to_agent():
            conn = win_openssh.OpenSSHAgentConnection()
        return conn
    else:
        return