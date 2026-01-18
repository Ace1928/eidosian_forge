import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_deleted_commit_message(self):
    self.make_uncommitted_tree()
    if sys.platform == 'win32':
        editor = 'cmd.exe /c del'
    else:
        editor = 'rm'
    self.overrideEnv('BRZ_EDITOR', editor)
    self.assertRaises((EnvironmentError, _mod_transport.NoSuchFile), msgeditor.edit_commit_message, '')