import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def fetch_class_stats(revs):
    return gather_class_stats(a_branch.repository, revs)