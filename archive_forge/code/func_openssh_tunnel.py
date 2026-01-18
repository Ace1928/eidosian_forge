import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def openssh_tunnel(lport, rport, server, remoteip='127.0.0.1', keyfile=None, password=None, timeout=60):
    """Create an ssh tunnel using command-line ssh that connects port lport
    on this machine to localhost:rport on server.  The tunnel
    will automatically close when not in use, remaining open
    for a minimum of timeout seconds for an initial connection.

    This creates a tunnel redirecting `localhost:lport` to `remoteip:rport`,
    as seen from `server`.

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
    if pexpect is None:
        raise ImportError('pexpect unavailable, use paramiko_tunnel')
    ssh = 'ssh '
    if keyfile:
        ssh += '-i ' + keyfile
    if ':' in server:
        server, port = server.split(':')
        ssh += ' -p %s' % port
    cmd = f'{ssh} -O check {server}'
    output, exitstatus = pexpect.run(cmd, withexitstatus=True)
    if not exitstatus:
        pid = int(output[output.find(b'(pid=') + 5:output.find(b')')])
        cmd = '%s -O forward -L 127.0.0.1:%i:%s:%i %s' % (ssh, lport, remoteip, rport, server)
        output, exitstatus = pexpect.run(cmd, withexitstatus=True)
        if not exitstatus:
            atexit.register(_stop_tunnel, cmd.replace('-O forward', '-O cancel', 1))
            return pid
    cmd = '%s -f -S none -L 127.0.0.1:%i:%s:%i %s sleep %i' % (ssh, lport, remoteip, rport, server, timeout)
    env = os.environ.copy()
    env.pop('SSH_ASKPASS', None)
    ssh_newkey = 'Are you sure you want to continue connecting'
    tunnel = pexpect.spawn(cmd, env=env)
    failed = False
    MAX_RETRY = 10
    for _ in range(MAX_RETRY):
        try:
            i = tunnel.expect([ssh_newkey, _password_pat], timeout=0.1)
            if i == 0:
                raise SSHException("The authenticity of the host can't be established.")
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            if tunnel.exitstatus:
                print(tunnel.exitstatus)
                print(tunnel.before)
                print(tunnel.after)
                raise RuntimeError("tunnel '%s' failed to start" % cmd)
            else:
                return tunnel.pid
        else:
            if failed:
                print('Password rejected, try again')
                password = None
            if password is None:
                password = getpass("%s's password: " % server)
            tunnel.sendline(password)
            failed = True
    raise MaxRetryExceeded(f'Failed after {MAX_RETRY} attempts')