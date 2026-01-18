import sys
import threading
import webbrowser
import socket
from http import server
from io import BytesIO as IO
import itertools
import random
def generate_handler(html, files=None):
    if files is None:
        files = {}

    class MyHandler(server.BaseHTTPRequestHandler):

        def do_GET(self):
            """Respond to a GET request."""
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path in files:
                content_type, content = files[self.path]
                self.send_response(200)
                self.send_header('Content-type', content_type)
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_error(404)
    return MyHandler