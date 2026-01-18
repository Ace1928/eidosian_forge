from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_incomplete_request(self):
    transport = FakeTransport()
    wsgi_app = wsgi.SmartWSGIApp(transport)

    def make_request(transport, write_func, bytes, root_client_path):
        request = IncompleteRequest(transport, write_func)
        request.accept_bytes(bytes)
        self.request = request
        return request
    wsgi_app.make_request = make_request
    fake_input = BytesIO(b'incomplete request')
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(fake_input.getvalue()), 'wsgi.input': fake_input, 'breezy.relpath': 'foo/bar'})
    iterable = wsgi_app(environ, self.start_response)
    response = self.read_response(iterable)
    self.assertEqual('200 OK', self.status)
    self.assertEqual(b'error\x01incomplete request\n', response)