import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
@unittest.skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
def testShortDescriptionWithMultiLineDocstring(self):
    """Tests shortDescription() for a method with a longer docstring.

        This method ensures that only the first line of a docstring is
        returned used in the short description, no matter how long the
        whole thing is.
        """
    self.assertEqual(self.shortDescription(), 'Tests shortDescription() for a method with a longer docstring.')