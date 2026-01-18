import base64
import re
from io import BytesIO
from urllib.request import parse_http_list, parse_keqv_list
from .. import errors, osutils, tests, transport
from ..bzr.smart import medium
from ..transport import chroot
from . import http_server
def do_POST_inner(self, chrooted_transport):
    self.send_response(200)
    self.send_header('Content-type', 'application/octet-stream')
    if not self.path.endswith('.bzr/smart'):
        raise AssertionError('POST to path not ending in .bzr/smart: {!r}'.format(self.path))
    t = chrooted_transport.clone(self.path[:-len('.bzr/smart')])
    data_length = int(self.headers['Content-Length'])
    request_bytes = self.rfile.read(data_length)
    protocol_factory, unused_bytes = medium._get_protocol_factory_for_bytes(request_bytes)
    out_buffer = BytesIO()
    smart_protocol_request = protocol_factory(t, out_buffer.write, '/')
    smart_protocol_request.accept_bytes(unused_bytes)
    if not smart_protocol_request.next_read_size() == 0:
        raise errors.SmartProtocolError('not finished reading, but all data sent to protocol.')
    self.send_header('Content-Length', str(len(out_buffer.getvalue())))
    self.end_headers()
    self.wfile.write(out_buffer.getvalue())