from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
def test_jail_root(self):
    """The WSGI HPSS glue allows access to the whole WSGI backing
        transport, regardless of which HTTP path the request was delivered
        to.
        """
    self.make_repository('repo', shared=True)
    branch = self.make_controldir('repo/branch').create_branch()
    wsgi_app = wsgi.SmartWSGIApp(self.get_transport())
    environ = self.make_hpss_wsgi_request('/repo/branch', b'BzrDir.open_branchV2', b'.')
    iterable = wsgi_app(environ, self.start_response)
    response_bytes = self.read_response(iterable)
    self.assertEqual('200 OK', self.status)
    from breezy.bzr.tests.test_smart_transport import LoggingMessageHandler
    message_handler = LoggingMessageHandler()
    decoder = protocol.ProtocolThreeDecoder(message_handler, expect_version_marker=True)
    decoder.accept_bytes(response_bytes)
    self.assertTrue(('structure', (b'branch', branch._format.network_name())) in message_handler.event_log)