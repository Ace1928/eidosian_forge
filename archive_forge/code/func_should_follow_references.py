import threading
from . import errors, trace, urlutils
from .branch import Branch
from .controldir import ControlDir, ControlDirFormat
from .transport import do_catching_redirections, get_transport
def should_follow_references(self):
    return True