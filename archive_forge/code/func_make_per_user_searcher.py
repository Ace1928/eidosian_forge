from breezy import rules
from breezy.tests.per_tree import TestCaseWithTree
def make_per_user_searcher(self, text):
    """Make a _RulesSearcher from a string"""
    return rules._IniBasedRulesSearcher(text.splitlines(True))