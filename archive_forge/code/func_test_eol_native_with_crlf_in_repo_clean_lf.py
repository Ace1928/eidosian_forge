import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def test_eol_native_with_crlf_in_repo_clean_lf(self):
    wt, basis = self.prepare_tree(_sample_clean_lf, eol='native-with-crlf-in-repo')
    self.assertContent(wt, basis, _sample_text_on_win, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_CRLF_IN_REPO)