from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
def test_different_name_case(self):
    revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Foo <foo@example.com>'), Revision('3', {}, committer='FOO <bar@example.com>')]
    foo = ('Foo', 'foo@example.com')
    FOO = ('FOO', 'bar@example.com')
    committers = {foo: foo, FOO: foo}
    info = collapse_by_person(revisions, committers)
    self.assertEqual(3, info[0][0])
    self.assertEqual({'foo@example.com': 2, 'bar@example.com': 1}, info[0][2])
    self.assertEqual({'Foo': 2, 'FOO': 1}, info[0][3])