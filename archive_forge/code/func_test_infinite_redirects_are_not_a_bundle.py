import socketserver
from .. import errors, tests
from ..bzr.tests import test_read_bundle
from ..directory_service import directories
from ..mergeable import read_mergeable_from_url
from . import test_server
def test_infinite_redirects_are_not_a_bundle(self):
    """If a URL causes TooManyRedirections then NotABundle is raised.
        """
    from .blackbox.test_push import RedirectingMemoryServer
    server = RedirectingMemoryServer()
    self.start_server(server)
    url = server.get_url() + 'infinite-loop'
    self.assertRaises(errors.NotABundle, read_mergeable_from_url, url)