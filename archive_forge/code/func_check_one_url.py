import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
def check_one_url(self, url):
    """Check that `url` is okay to open."""
    if urlutils.URL.from_string(str(url)).scheme != self.allowed_scheme:
        raise BadUrl(url)