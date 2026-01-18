from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_http_get_rejected(self):
    app = wsgi.SmartWSGIApp(FakeTransport())
    environ = self.build_environ({'REQUEST_METHOD': 'GET'})
    iterable = app(environ, self.start_response)
    self.read_response(iterable)
    self.assertEqual('405 Method not allowed', self.status)
    self.assertTrue(('Allow', 'POST') in self.headers)