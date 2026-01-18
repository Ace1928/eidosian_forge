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
def args_received(self, args):
    cmd = args[0]
    args = args[1:]
    try:
        command = self._commands.get(cmd)
    except LookupError:
        if 'hpss' in debug.debug_flags:
            self._trace('hpss unknown request', cmd, repr(args)[1:-1])
        raise errors.UnknownSmartMethod(cmd)
    if 'hpss' in debug.debug_flags:
        from . import vfs
        if issubclass(command, vfs.VfsRequest):
            action = 'hpss vfs req'
        else:
            action = 'hpss request'
        self._trace(action, '{} {}'.format(cmd, repr(args)[1:-1]))
    self._command = command(self._backing_transport, self._root_client_path, self._jail_root)
    self._run_handler_code(self._command.execute, args, {})