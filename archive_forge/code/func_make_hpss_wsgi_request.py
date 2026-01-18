from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def make_hpss_wsgi_request(self, wsgi_relpath, *args):
    write_buf = BytesIO()
    request_medium = medium.SmartSimplePipesClientMedium(None, write_buf, 'fake:' + wsgi_relpath)
    request_encoder = protocol.ProtocolThreeRequester(request_medium.get_request())
    request_encoder.call(*args)
    write_buf.seek(0)
    environ = self.build_environ({'REQUEST_METHOD': 'POST', 'CONTENT_LENGTH': len(write_buf.getvalue()), 'wsgi.input': write_buf, 'breezy.relpath': wsgi_relpath})
    return environ