import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
def test_asterisk(self):
    self.assertMatch([('x*x', ['xx', 'x.x', 'x茶..x', '茶/x.x', 'x.y.x'], ['x/x', 'bar/x/bar/x', 'bax/abaxab']), ('foo/*x', ['foo/x', 'foo/bax', 'foo/a.x', 'foo/.x', 'foo/.q.x'], ['foo/bar/bax']), ('*/*x', ['茶/x', 'foo/x', 'foo/bax', 'x/a.x', '.foo/x', '茶/.x', 'foo/.q.x'], ['foo/bar/bax']), ('f*', ['foo', 'foo.bar'], ['.foo', 'foo/bar', 'foo/.bar']), ('*bar', ['bar', 'foobar', 'foo\\nbar', 'foo.bar', 'foo/bar', 'foo/foobar', 'foo/f.bar', '.bar', 'foo/.bar'], [])])