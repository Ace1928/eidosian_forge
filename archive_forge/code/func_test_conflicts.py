from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_conflicts(self):
    wt = self.make_branch_and_tree('old')
    wt.commit('base', rev_id=b'base')
    self.build_tree(['old/afile'])
    wt.add(['afile'], ids=[b'originalid'])
    wt.commit('bla', rev_id=b'oldparent')
    with open('old/afile', 'w') as f:
        f.write('bloe')
    wt.commit('bla', rev_id=b'oldcommit')
    oldrepos = wt.branch.repository
    wt = self.make_branch_and_tree('new')
    self.build_tree(['new/afile'])
    wt.add(['afile'], ids=[b'newid'])
    wt.commit('bla', rev_id=b'newparent')
    wt.branch.repository.fetch(oldrepos)
    with wt.lock_write():
        replayer = WorkingTreeRevisionRewriter(wt, RebaseState1(wt))
        self.assertRaises(ConflictsInTree, replayer, b'oldcommit', b'newcommit', [b'newparent'])