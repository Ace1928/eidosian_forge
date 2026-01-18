import time
from . import debug, errors, osutils, revision, trace
def find_difference(self, left_revision, right_revision):
    """Determine the graph difference between two revisions"""
    border, common, searchers = self._find_border_ancestors([left_revision, right_revision])
    self._search_for_extra_common(common, searchers)
    left = searchers[0].seen
    right = searchers[1].seen
    return (left.difference(right), right.difference(left))