import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
def suite():
    from genshi.input import HTML
    from genshi.core import Markup
    from genshi.builder import tag
    suite = unittest.TestSuite()
    for test in (SelectTest, InvertTest, EndTest, EmptyTest, RemoveTest, UnwrapText, WrapTest, FilterTest, MapTest, SubstituteTest, RenameTest, ReplaceTest, BeforeTest, AfterTest, PrependTest, AppendTest, AttrTest, CopyTest, CutTest):
        suite.addTest(unittest.makeSuite(test, 'test'))
    suite.addTest(doctest_suite(genshi.filters.transform, optionflags=doctest.NORMALIZE_WHITESPACE, extraglobs={'HTML': HTML, 'tag': tag, 'Markup': Markup}))
    return suite