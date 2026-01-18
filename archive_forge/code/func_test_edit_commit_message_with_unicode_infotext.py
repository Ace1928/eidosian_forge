import os
import sys
from .. import commit, config, errors, msgeditor, osutils, trace
from .. import transport as _mod_transport
from ..msgeditor import (edit_commit_message_encoded,
from ..trace import mutter
from . import (TestCaseInTempDir, TestCaseWithTransport, TestNotApplicable,
from .EncodingAdapter import encoding_scenarios
import sys
def test_edit_commit_message_with_unicode_infotext(self):
    self.make_uncommitted_tree()
    self.make_fake_editor()
    mutter('edit_commit_message with unicode infotext')
    uni_val, ue_val = probe_unicode_in_user_encoding()
    if ue_val is None:
        self.skipTest('Cannot find a unicode character that works in encoding %s' % (osutils.get_user_encoding(),))
    self.assertEqual('test message from fed\n', msgeditor.edit_commit_message(uni_val))
    tmpl = edit_commit_message_encoded('ሴ'.encode())
    self.assertEqual('test message from fed\n', tmpl)