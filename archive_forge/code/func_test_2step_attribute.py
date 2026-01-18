import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_2step_attribute(self):
    xml = XML('<elem class="x"><span id="joe">Hey Joe</span></elem>')
    self._test_eval('@*', input=xml, output='x')
    self._test_eval('./@*', input=xml, output='x')
    self._test_eval('.//@*', input=xml, output='xjoe')
    self._test_eval('*/@*', input=xml, output='joe')
    xml = XML('<elem><foo id="1"/><foo id="2"/></elem>')
    self._test_eval('@*', input=xml, output='')
    self._test_eval('foo/@*', input=xml, output='12')