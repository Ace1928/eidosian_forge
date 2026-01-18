import ssl
import tempfile
import threading
import pytest
from requests.compat import urljoin
@pytest.fixture
def nosan_server(tmp_path_factory):
    import trustme
    tmpdir = tmp_path_factory.mktemp('certs')
    ca = trustme.CA()
    server_cert = ca.issue_cert(common_name=u'localhost')
    ca_bundle = str(tmpdir / 'ca.pem')
    ca.cert_pem.write_to_path(ca_bundle)
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_cert.configure_cert(context)
    server = HTTPServer(('localhost', 0), SimpleHTTPRequestHandler)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    yield ('localhost', server.server_address[1], ca_bundle)
    server.shutdown()
    server_thread.join()