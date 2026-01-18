import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test__run_editor_EACCES(self):
    """If running a configured editor raises EACESS, the user is warned."""
    self.overrideEnv('BRZ_EDITOR', 'eacces.py')
    with open('eacces.py', 'wb') as f:
        f.write(b'# Not a real editor')
    os.chmod('eacces.py', 0)
    self.overrideEnv('EDITOR', self.make_do_nothing_editor())
    warnings = []

    def warning(*args):
        if len(args) > 1:
            warnings.append(args[0] % args[1:])
        else:
            warnings.append(args[0])
    _warning = trace.warning
    trace.warning = warning
    try:
        msgeditor._run_editor('')
    finally:
        trace.warning = _warning
    self.assertStartsWith(warnings[0], 'Could not start editor "eacces.py"')