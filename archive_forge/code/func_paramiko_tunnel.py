import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def paramiko_tunnel(lport, rport, server, remoteip='127.0.0.1', keyfile=None, password=None, timeout=60):
    """launch a tunner with paramiko in a subprocess. This should only be used
    when shell ssh is unavailable (e.g. Windows).

    This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
    as seen from `server`.

    If you are familiar with ssh tunnels, this creates the tunnel:

    ssh server -L localhost:lport:remoteip:rport

    keyfile and password may be specified, but ssh config is checked for defaults.


    Parameters
    ----------

    lport : int
        local port for connecting to the tunnel from this machine.
    rport : int
        port on the remote machine to connect to.
    server : str
        The ssh server to connect to. The full ssh server string will be parsed.
        user@server:port
    remoteip : str [Default: 127.0.0.1]
        The remote ip, specifying the destination of the tunnel.
        Default is localhost, which means that the tunnel would redirect
        localhost:lport on this machine to localhost:rport on the *server*.

    keyfile : str; path to private key file
        This specifies a key to be used in ssh login, default None.
        Regular default ssh keys will be used without specifying this argument.
    password : str;
        Your ssh password to the ssh server. Note that if this is left None,
        you will be prompted for it if passwordless key based login is unavailable.
    timeout : int [default: 60]
        The time (in seconds) after which no activity will result in the tunnel
        closing.  This prevents orphaned tunnels from running forever.

    """
    if paramiko is None:
        raise ImportError('Paramiko not available')
    if password is None:
        if not _try_passwordless_paramiko(server, keyfile):
            password = getpass("%s's password: " % server)
    p = Process(target=_paramiko_tunnel, args=(lport, rport, server, remoteip), kwargs=dict(keyfile=keyfile, password=password))
    p.daemon = True
    p.start()
    return p