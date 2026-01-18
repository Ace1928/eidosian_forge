from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def is_mq(fileobj, path=None):
    fd = _convert_fileobj(fileobj)
    return _is_mq(fd, path)