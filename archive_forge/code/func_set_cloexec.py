import logging
from logging import NullHandler
from vine import promise  # noqa
from vine.utils import wraps
def set_cloexec(fd, cloexec):
    """Set flag to close fd after exec."""
    if fcntl is None:
        return
    try:
        FD_CLOEXEC = fcntl.FD_CLOEXEC
    except AttributeError:
        raise NotImplementedError('close-on-exec flag not supported on this platform')
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    if cloexec:
        flags |= FD_CLOEXEC
    else:
        flags &= ~FD_CLOEXEC
    return fcntl.fcntl(fd, fcntl.F_SETFD, flags)