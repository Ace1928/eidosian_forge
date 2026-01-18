import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def make_fake_editor(self, message='test message from fed\n'):
    """Set up environment so that an editor will be a known script.

        Sets up BRZ_EDITOR so that if an editor is spawned it will run a
        script that just adds a known message to the start of the file.
        """
    if not isinstance(message, bytes):
        message = message.encode('utf-8')
    with open('fed.py', 'w') as f:
        f.write('#!%s\n' % sys.executable)
        f.write("# coding=utf-8\nimport sys\nif len(sys.argv) == 2:\n    fn = sys.argv[1]\n    with open(fn, 'rb') as f:\n        s = f.read()\n    with open(fn, 'wb') as f:\n        f.write({!r})\n        f.write(s)\n".format(message))
    if sys.platform == 'win32':
        with open('fed.bat', 'w') as f:
            f.write('@echo off\n"%s" fed.py %%1\n' % sys.executable)
        self.overrideEnv('BRZ_EDITOR', 'fed.bat')
    else:
        os.chmod('fed.py', 493)
        mutter('Setting BRZ_EDITOR to %r', '%s ./fed.py' % sys.executable)
        self.overrideEnv('BRZ_EDITOR', '%s ./fed.py' % sys.executable)