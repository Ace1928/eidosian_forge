from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def is_socket(fileobj, family=_AF_UNSPEC, type=0, listening=-1):
    fd = _convert_fileobj(fileobj)
    return _is_socket(fd, family, type, listening)