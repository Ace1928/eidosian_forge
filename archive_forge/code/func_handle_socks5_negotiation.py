import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def handle_socks5_negotiation(sock, negotiate, username=None, password=None):
    """
    Handle the SOCKS5 handshake.

    Returns a generator object that allows us to break the handshake into
    steps so that the test code can intervene at certain useful points.
    """
    received_version = sock.recv(1)
    assert received_version == SOCKS_VERSION_SOCKS5
    nmethods = ord(sock.recv(1))
    methods = _read_exactly(sock, nmethods)
    if negotiate:
        assert SOCKS_NEGOTIATION_PASSWORD in methods
        send_data = SOCKS_VERSION_SOCKS5 + SOCKS_NEGOTIATION_PASSWORD
        sock.sendall(send_data)
        negotiation_version = sock.recv(1)
        assert negotiation_version == b'\x01'
        ulen = ord(sock.recv(1))
        provided_username = _read_exactly(sock, ulen)
        plen = ord(sock.recv(1))
        provided_password = _read_exactly(sock, plen)
        if username == provided_username and password == provided_password:
            sock.sendall(b'\x01\x00')
        else:
            sock.sendall(b'\x01\x01')
            sock.close()
            yield False
            return
    else:
        assert SOCKS_NEGOTIATION_NONE in methods
        send_data = SOCKS_VERSION_SOCKS5 + SOCKS_NEGOTIATION_NONE
        sock.sendall(send_data)
    received_version = sock.recv(1)
    command = sock.recv(1)
    reserved = sock.recv(1)
    addr = _address_from_socket(sock)
    port = _read_exactly(sock, 2)
    port = (ord(port[0:1]) << 8) + ord(port[1:2])
    assert received_version == SOCKS_VERSION_SOCKS5
    assert command == b'\x01'
    assert reserved == b'\x00'
    succeed = (yield (addr, port))
    if succeed:
        response = SOCKS_VERSION_SOCKS5 + b'\x00\x00\x01\x7f\x00\x00\x01\xea`'
    else:
        response = SOCKS_VERSION_SOCKS5 + b'\x01\x00'
    sock.sendall(response)
    yield True