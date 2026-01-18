import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test__create_temp_file_with_commit_template_in_unicode_dir(self):
    self.requireFeature(features.UnicodeFilenameFeature)
    if hasattr(self, 'info'):
        tmpdir = self.info['directory']
        os.mkdir(tmpdir)
        msgeditor._create_temp_file_with_commit_template(b'infotext', tmpdir=tmpdir)
    else:
        raise TestNotApplicable('Test run elsewhere with non-ascii data.')