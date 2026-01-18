from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
def create_editor_hook(pydev_host, pydev_client_port):

    def call_editor(filename, line=0, wait=True):
        """ Open an editor in PyDev """
        if line is None:
            line = 0
        filename = os.path.abspath(filename)
        server = xmlrpclib.Server('http://%s:%s' % (pydev_host, pydev_client_port))
        server.IPythonEditor(filename, str(line))
        if wait:
            input('Press Enter when done editing:')
    return call_editor