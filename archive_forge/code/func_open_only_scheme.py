import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
def open_only_scheme(allowed_scheme, url):
    """Open the branch at `url`, only accessing URLs on `allowed_scheme`.

    :raises BadUrl: An attempt was made to open a URL that was not on
        `allowed_scheme`.
    """
    return BranchOpener(SingleSchemePolicy(allowed_scheme)).open(url)