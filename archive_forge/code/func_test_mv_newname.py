import os
from ... import osutils, tests
from ...osutils import canonical_relpath, pathjoin
from .. import KnownFailure
from ..features import CaseInsCasePresFilenameFeature
from ..script import run_script
def test_mv_newname(self):
    wt = self._make_mixed_case_tree()
    run_script(self, '\n            $ brz add -q\n            $ brz ci -qm message\n            $ brz mv camelcaseparent/camelcase camelcaseparent/NewCamelCase\n            CamelCaseParent/CamelCase => CamelCaseParent/NewCamelCase\n            ')