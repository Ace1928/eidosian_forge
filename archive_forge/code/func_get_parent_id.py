from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
def get_parent_id(t, p):
    if p:
        return t.path2id(osutils.dirname(p))
    else:
        return None