from _pydev_bundle._pydev_saved_modules import socket
import sys
def get_socket_names(n_sockets, close=False):
    socket_names = []
    sockets = []
    for _ in range(n_sockets):
        if IS_JYTHON:
            from java.net import ServerSocket
            sock = ServerSocket(0)
            socket_name = (get_localhost(), sock.getLocalPort())
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((get_localhost(), 0))
            socket_name = sock.getsockname()
        sockets.append(sock)
        socket_names.append(socket_name)
    if close:
        for s in sockets:
            s.close()
    return socket_names