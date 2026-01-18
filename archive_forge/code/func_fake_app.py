from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def fake_app(environ, start_response):
    self.fail('The app should never be called when the path is wrong')