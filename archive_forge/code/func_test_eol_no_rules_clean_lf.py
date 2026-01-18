import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def test_eol_no_rules_clean_lf(self):
    wt, basis = self.prepare_tree(_sample_clean_lf)
    self.assertContent(wt, basis, _sample_clean_lf, _sample_text_on_unix, _sample_text_on_win, roundtrip_to=_LF_IN_REPO)