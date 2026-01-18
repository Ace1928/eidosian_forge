import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def finish_brz_subprocess_with_import_check(self, process, args, forbidden_imports):
    """Finish subprocess and check specific modules have not been
        imported.

        :param forbidden_imports: List of fully-qualified Python module names
            that should not be loaded while running this command.
        """
    out, err = self.finish_brz_subprocess(process, universal_newlines=False, process_args=args)
    self.check_forbidden_modules(err, forbidden_imports)
    return (out, err)