from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_make_app(self):
    app = wsgi.make_app(root='a root', prefix='a prefix', path_var='a path_var')
    self.assertIsInstance(app, wsgi.RelpathSetter)
    self.assertIsInstance(app.app, wsgi.SmartWSGIApp)
    self.assertStartsWith(app.app.backing_transport.base, 'chroot-')
    backing_transport = app.app.backing_transport
    chroot_backing_transport = backing_transport.server.backing_transport
    self.assertEndsWith(chroot_backing_transport.base, 'a%20root/')
    self.assertEqual(app.app.root_client_path, 'a prefix')
    self.assertEqual(app.path_var, 'a path_var')