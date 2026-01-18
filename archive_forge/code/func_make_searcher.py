import sys
from breezy import rules, tests
def make_searcher(self, text1=None, text2=None):
    """Make a _StackedRulesSearcher with 0, 1 or 2 items"""
    searchers = []
    if text1 is not None:
        searchers.append(rules._IniBasedRulesSearcher(text1.splitlines()))
    if text2 is not None:
        searchers.append(rules._IniBasedRulesSearcher(text2.splitlines()))
    return rules._StackedRulesSearcher(searchers)