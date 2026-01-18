import io
import re
import socket
from gunicorn.http.body import ChunkedReader, LengthReader, EOFReader, Body
from gunicorn.http.errors import (
from gunicorn.http.errors import InvalidProxyLine, ForbiddenProxyRequest
from gunicorn.http.errors import InvalidSchemeHeaders
from gunicorn.util import bytes_to_str, split_request_uri
def parse_proxy_protocol(self, line):
    bits = line.split()
    if len(bits) != 6:
        raise InvalidProxyLine(line)
    proto = bits[1]
    s_addr = bits[2]
    d_addr = bits[3]
    if proto not in ['TCP4', 'TCP6']:
        raise InvalidProxyLine("protocol '%s' not supported" % proto)
    if proto == 'TCP4':
        try:
            socket.inet_pton(socket.AF_INET, s_addr)
            socket.inet_pton(socket.AF_INET, d_addr)
        except socket.error:
            raise InvalidProxyLine(line)
    elif proto == 'TCP6':
        try:
            socket.inet_pton(socket.AF_INET6, s_addr)
            socket.inet_pton(socket.AF_INET6, d_addr)
        except socket.error:
            raise InvalidProxyLine(line)
    try:
        s_port = int(bits[4])
        d_port = int(bits[5])
    except ValueError:
        raise InvalidProxyLine('invalid port %s' % line)
    if not (0 <= s_port <= 65535 and 0 <= d_port <= 65535):
        raise InvalidProxyLine('invalid port %s' % line)
    self.proxy_protocol_info = {'proxy_protocol': proto, 'client_addr': s_addr, 'client_port': s_port, 'proxy_addr': d_addr, 'proxy_port': d_port}