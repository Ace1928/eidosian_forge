import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_basic_waiting_server(self):
    """the server waits for the block_server event to be set before closing"""
    block_server = threading.Event()
    with Server.basic_response_server(wait_to_close_event=block_server) as (host, port):
        sock = socket.socket()
        sock.connect((host, port))
        sock.sendall(b'send something')
        time.sleep(2.5)
        sock.sendall(b'still alive')
        block_server.set()