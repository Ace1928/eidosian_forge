import codecs
from io import BytesIO, StringIO
from .. import annotate, tests
from .ui_testing import StringIOWithEncoding
def test_annotate_author_or_committer(self):
    tree1 = self.make_branch_and_tree('tree1')
    self.build_tree_contents([('tree1/a', b'hello')])
    tree1.add(['a'], ids=[b'a-id'])
    tree1.commit('a', rev_id=b'rev-1', committer='Committer <committer@example.com>', timestamp=1166046000.0, timezone=0)
    self.build_tree_contents([('tree1/b', b'bye')])
    tree1.add(['b'], ids=[b'b-id'])
    tree1.commit('b', rev_id=b'rev-2', committer='Committer <committer@example.com>', authors=['Author <author@example.com>'], timestamp=1166046000.0, timezone=0)
    tree1.lock_read()
    self.addCleanup(tree1.unlock)
    self.assertBranchAnnotate('1   committ | hello\n', tree1.branch, 'a', b'rev-1')
    self.assertBranchAnnotate('2   author@ | bye\n', tree1.branch, 'b', b'rev-2')