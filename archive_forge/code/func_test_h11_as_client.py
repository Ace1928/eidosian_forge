import json
import os.path
import socket
import socketserver
import threading
from contextlib import closing, contextmanager
from http.server import SimpleHTTPRequestHandler
from typing import Callable, Generator
from urllib.request import urlopen
import h11
def test_h11_as_client() -> None:
    with socket_server(SingleMindedRequestHandler) as httpd:
        with closing(socket.create_connection(httpd.server_address)) as s:
            c = h11.Connection(h11.CLIENT)
            s.sendall(c.send(h11.Request(method='GET', target='/foo', headers=[('Host', 'localhost')])))
            s.sendall(c.send(h11.EndOfMessage()))
            data = bytearray()
            while True:
                event = c.next_event()
                print(event)
                if event is h11.NEED_DATA:
                    c.receive_data(s.recv(10))
                    continue
                if type(event) is h11.Response:
                    assert event.status_code == 200
                if type(event) is h11.Data:
                    data += event.data
                if type(event) is h11.EndOfMessage:
                    break
            assert bytes(data) == test_file_data