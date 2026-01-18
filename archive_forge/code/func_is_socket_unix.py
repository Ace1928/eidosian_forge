from socket import AF_UNSPEC as _AF_UNSPEC
from ._daemon import (__version__,
def is_socket_unix(fileobj, type=0, listening=-1, path=None):
    fd = _convert_fileobj(fileobj)
    return _is_socket_unix(fd, type, listening, path)