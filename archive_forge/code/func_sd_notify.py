import os
import socket
def sd_notify(state, logger, unset_environment=False):
    """Send a notification to systemd. state is a string; see
    the man page of sd_notify (http://www.freedesktop.org/software/systemd/man/sd_notify.html)
    for a description of the allowable values.

    If the unset_environment parameter is True, sd_notify() will unset
    the $NOTIFY_SOCKET environment variable before returning (regardless of
    whether the function call itself succeeded or not). Further calls to
    sd_notify() will then fail, but the variable is no longer inherited by
    child processes.
    """
    addr = os.environ.get('NOTIFY_SOCKET')
    if addr is None:
        return
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC)
        if addr[0] == '@':
            addr = '\x00' + addr[1:]
        sock.connect(addr)
        sock.sendall(state.encode('utf-8'))
    except Exception:
        logger.debug('Exception while invoking sd_notify()', exc_info=True)
    finally:
        if unset_environment:
            os.environ.pop('NOTIFY_SOCKET')
        sock.close()