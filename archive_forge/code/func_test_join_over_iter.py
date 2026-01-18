import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_join_over_iter(self):
    items = ['foo', '<bar />', Markup('<baz />')]
    markup = Markup('<br />').join((i for i in items))
    self.assertEqual('foo<br />&lt;bar /&gt;<br /><baz />', markup)