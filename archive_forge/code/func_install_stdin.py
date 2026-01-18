import threading
import sys
from paste.util import filemixin
def install_stdin(**kw):
    global _stdincatcher, _oldstdin, register_stdin, deregister_stdin
    if not _stdincatcher:
        _oldstdin = sys.stdin
        _stdincatcher = sys.stdin = StdinCatcher(**kw)
        register_stdin = _stdincatcher.register
        deregister_stdin = _stdincatcher.deregister