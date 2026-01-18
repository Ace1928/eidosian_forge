import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_2step_complex(self):
    xml = XML('<root><foo><bar/></foo></root>')
    self._test_eval(path='foo/bar', equiv='<Path "child::foo/child::bar">', input=xml, output='<bar/>')
    self._test_eval(path='./bar', equiv='<Path "self::node()/child::bar">', input=xml, output='')
    self._test_eval(path='foo/*', equiv='<Path "child::foo/child::*">', input=xml, output='<bar/>')
    xml = XML('<root><foo><bar id="1"/></foo><bar id="2"/></root>')
    self._test_eval(path='./bar', equiv='<Path "self::node()/child::bar">', input=xml, output='<bar id="2"/>')
    xml = XML('<table>\n            <tr><td>1</td><td>One</td></tr>\n            <tr><td>2</td><td>Two</td></tr>\n        </table>')
    self._test_eval(path='tr/td[1]', input=xml, output='<td>1</td><td>2</td>')
    xml = XML('<ul>\n            <li>item1\n                <ul><li>subitem11</li></ul>\n            </li>\n            <li>item2\n                <ul><li>subitem21</li></ul>\n            </li>\n        </ul>')
    self._test_eval(path='li[2]/ul', input=xml, output='<ul><li>subitem21</li></ul>')