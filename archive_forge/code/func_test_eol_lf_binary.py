import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def test_eol_lf_binary(self):
    wt, basis = self.prepare_tree(_sample_binary, eol='lf')
    self.assertContent(wt, basis, _sample_binary, _sample_binary, _sample_binary)