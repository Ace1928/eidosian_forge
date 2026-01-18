import errno
import os
import os.path
import random
import socket
import sys
import ovs.fatal_signal
import ovs.poller
import ovs.vlog
def make_unix_socket(style, nonblock, bind_path, connect_path, short=False):
    """Creates a Unix domain socket in the given 'style' (either
    socket.SOCK_DGRAM or socket.SOCK_STREAM) that is bound to 'bind_path' (if
    'bind_path' is not None) and connected to 'connect_path' (if 'connect_path'
    is not None).  If 'nonblock' is true, the socket is made non-blocking.

    Returns (error, socket): on success 'error' is 0 and 'socket' is a new
    socket object, on failure 'error' is a positive errno value and 'socket' is
    None."""
    try:
        sock = socket.socket(socket.AF_UNIX, style)
    except socket.error as e:
        return (get_exception_errno(e), None)
    try:
        if nonblock:
            set_nonblocking(sock)
        if bind_path is not None:
            try:
                os.unlink(bind_path)
            except OSError as e:
                if e.errno != errno.ENOENT:
                    return (e.errno, None)
            ovs.fatal_signal.add_file_to_unlink(bind_path)
            sock.bind(bind_path)
            try:
                os.fchmod(sock.fileno(), 448)
            except OSError:
                pass
        if connect_path is not None:
            try:
                sock.connect(connect_path)
            except socket.error as e:
                if get_exception_errno(e) != errno.EINPROGRESS:
                    raise
        return (0, sock)
    except socket.error as e:
        sock.close()
        if bind_path is not None and os.path.exists(bind_path):
            ovs.fatal_signal.unlink_file_now(bind_path)
        eno = ovs.socket_util.get_exception_errno(e)
        if eno == 'AF_UNIX path too long' and os.uname()[0] == 'Linux':
            short_connect_path = None
            short_bind_path = None
            connect_dirfd = None
            bind_dirfd = None
            if connect_path is not None:
                dirname = os.path.dirname(connect_path)
                basename = os.path.basename(connect_path)
                try:
                    connect_dirfd = os.open(dirname, os.O_DIRECTORY | os.O_RDONLY)
                except OSError as err:
                    return (get_exception_errno(err), None)
                short_connect_path = '/proc/self/fd/%d/%s' % (connect_dirfd, basename)
            if bind_path is not None:
                dirname = os.path.dirname(bind_path)
                basename = os.path.basename(bind_path)
                try:
                    bind_dirfd = os.open(dirname, os.O_DIRECTORY | os.O_RDONLY)
                except OSError as err:
                    return (get_exception_errno(err), None)
                short_bind_path = '/proc/self/fd/%d/%s' % (bind_dirfd, basename)
            try:
                return make_unix_socket(style, nonblock, short_bind_path, short_connect_path)
            finally:
                if connect_dirfd is not None:
                    os.close(connect_dirfd)
                if bind_dirfd is not None:
                    os.close(bind_dirfd)
        elif eno == 'AF_UNIX path too long':
            if short:
                return (get_exception_errno(e), None)
            short_bind_path = None
            try:
                short_bind_path = make_short_name(bind_path)
                short_connect_path = make_short_name(connect_path)
            except:
                free_short_name(short_bind_path)
                return (errno.ENAMETOOLONG, None)
            try:
                return make_unix_socket(style, nonblock, short_bind_path, short_connect_path, short=True)
            finally:
                free_short_name(short_bind_path)
                free_short_name(short_connect_path)
        else:
            return (get_exception_errno(e), None)