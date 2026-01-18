from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_generate_transpose_plan(self):
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    with open('hello', 'w') as f:
        f.write('hello world')
    wt.add('hello')
    wt.commit(message='add hello', rev_id=b'bla')
    with open('hello', 'w') as f:
        f.write('world')
    wt.commit(message='change hello', rev_id=b'bloe')
    wt.set_last_revision(b'bla')
    b.generate_revision_history(b'bla')
    with open('hello', 'w') as f:
        f.write('world')
    wt.commit(message='change hello', rev_id=b'bla2')
    with open('hello', 'w') as f:
        f.write('universe')
    wt.commit(message='change hello again', rev_id=b'bla3')
    wt.set_last_revision(b'bla')
    b.generate_revision_history(b'bla')
    with open('hello', 'w') as f:
        f.write('somebar')
    wt.commit(message='change hello yet again', rev_id=b'blie')
    wt.set_last_revision(NULL_REVISION)
    b.generate_revision_history(NULL_REVISION)
    wt.add('hello')
    wt.commit(message='add hello', rev_id=b'lala')
    b.repository.lock_read()
    graph = b.repository.get_graph()
    self.assertEqual({b'blie': (b'newblie', (b'lala',))}, generate_transpose_plan(graph.iter_ancestry([b'blie']), {b'bla': b'lala'}, graph, lambda y, _: b'new' + y))
    self.assertEqual({b'bla2': (b'newbla2', (b'lala',)), b'bla3': (b'newbla3', (b'newbla2',)), b'blie': (b'newblie', (b'lala',)), b'bloe': (b'newbloe', (b'lala',))}, generate_transpose_plan(graph.iter_ancestry(b.repository._all_revision_ids()), {b'bla': b'lala'}, graph, lambda y, _: b'new' + y))
    b.repository.unlock()