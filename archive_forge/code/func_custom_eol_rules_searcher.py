import sys
from io import BytesIO
from ... import rules, status
from ...workingtree import WorkingTree
from .. import TestSkipped
from . import TestCaseWithWorkingTree
def custom_eol_rules_searcher(tree, default_searcher):
    return rules._IniBasedRulesSearcher(['[name *]\n', 'eol=%s\n' % eol])