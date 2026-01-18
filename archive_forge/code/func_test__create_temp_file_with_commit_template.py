import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test__create_temp_file_with_commit_template(self):
    create_file = msgeditor._create_temp_file_with_commit_template
    msgfilename, hasinfo = create_file(b'infotext', '----', b'start message')
    self.assertNotEqual(None, msgfilename)
    self.assertTrue(hasinfo)
    expected = os.linesep.join(['start message', '', '', '----', '', 'infotext'])
    self.assertFileEqual(expected, msgfilename)