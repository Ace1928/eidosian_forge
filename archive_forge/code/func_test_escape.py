import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_escape(self):
    markup = escape('<b>"&"</b>')
    assert type(markup) is Markup
    self.assertEqual('&lt;b&gt;&#34;&amp;&#34;&lt;/b&gt;', markup)