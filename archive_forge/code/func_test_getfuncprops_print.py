import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
@unittest.skipIf(pypy, "pypy builtin signatures aren't complete")
def test_getfuncprops_print(self):
    props = inspection.getfuncprops('print', print)
    self.assertEqual(props.func, 'print')
    self.assertIn('end', props.argspec.kwonly)
    self.assertIn('file', props.argspec.kwonly)
    self.assertIn('flush', props.argspec.kwonly)
    self.assertIn('sep', props.argspec.kwonly)
    if _is_py311:
        self.assertEqual(repr(props.argspec.kwonly_defaults['file']), 'None')
    else:
        self.assertEqual(repr(props.argspec.kwonly_defaults['file']), 'sys.stdout')
    self.assertEqual(repr(props.argspec.kwonly_defaults['flush']), 'False')