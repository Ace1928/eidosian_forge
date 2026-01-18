from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_noAttrs(self):
    """
        L{elementMaker} should render an element with no attributes.
        """
    expected = '<div ></div>'
    self.assertEqual(expected, self.elementMaker('div'))