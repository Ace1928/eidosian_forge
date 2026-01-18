import threading
import sys
from paste.util import filemixin
def not_installed_error(*args, **kw):
    assert False, 'threadedprint has not yet been installed (call threadedprint.install())'