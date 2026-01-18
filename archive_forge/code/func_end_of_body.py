import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
def end_of_body(self):
    """No more body data will be received."""
    self._run_handler_code(self._command.do_end, (), {})
    self.finished_reading = True
    if 'hpss' in debug.debug_flags:
        self._trace('end of body', '', include_time=True)