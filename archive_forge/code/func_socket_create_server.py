from __future__ import annotations
def socket_create_server(address, family=socket.AF_INET):
    """Simplified backport of socket.create_server from Python 3.8."""
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind(address)
        sock.listen()
        return sock
    except socket.error:
        sock.close()
        raise