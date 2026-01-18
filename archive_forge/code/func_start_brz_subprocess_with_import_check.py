import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def start_brz_subprocess_with_import_check(self, args, stderr_file=None):
    """Run a bzr process and capture the imports.

        This is fairly expensive because we start a subprocess, so we aim to
        cover representative rather than exhaustive cases.
        """
    env_changes = dict(PYTHONVERBOSE='1', **self.preserved_env_vars)
    trace.mutter('Setting env for bzr subprocess: %r', env_changes)
    kwargs = dict(env_changes=env_changes, allow_plugins=False)
    if stderr_file:
        kwargs['stderr'] = stderr_file
    return self.start_brz_subprocess(args, **kwargs)