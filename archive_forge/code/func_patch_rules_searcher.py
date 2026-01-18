import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def patch_rules_searcher(self, eol):
    """Patch in a custom rules searcher with a given eol setting."""
    if eol is None:
        WorkingTree._get_rules_searcher = self.real_rules_searcher
    else:

        def custom_eol_rules_searcher(tree, default_searcher):
            return rules._IniBasedRulesSearcher(['[name *]\n', 'eol=%s\n' % eol])
        WorkingTree._get_rules_searcher = custom_eol_rules_searcher