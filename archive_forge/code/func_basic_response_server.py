import threading
import socket
import select
@classmethod
def basic_response_server(cls, **kwargs):
    return cls.text_response_server('HTTP/1.1 200 OK\r\n' + 'Content-Length: 0\r\n\r\n', **kwargs)