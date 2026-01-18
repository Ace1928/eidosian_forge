from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
def test_different_name(self):
    revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Bar <foo@example.com>'), Revision('3', {}, committer='Bar <foo@example.com>')]
    foo = ('Foo', 'foo@example.com')
    bar = ('Bar', 'foo@example.com')
    committers = {foo: foo, bar: foo}
    info = collapse_by_person(revisions, committers)
    self.assertEqual(3, info[0][0])
    self.assertEqual({'foo@example.com': 3}, info[0][2])
    self.assertEqual({'Foo': 1, 'Bar': 2}, info[0][3])