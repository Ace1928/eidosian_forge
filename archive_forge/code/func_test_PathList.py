import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def test_PathList(self):

    def norm(x):
        if cwd[1] == ':' and x[0] == '/':
            x = cwd[:2] + x
        return x.replace('/', os.path.sep)
    cwd = os.getcwd() + os.path.sep
    c = ConfigDict()
    self.assertEqual(PathList().domain_name(), 'PathList')
    c.declare('a', ConfigValue(None, PathList()))
    self.assertEqual(c.a, None)
    c.a = '/a/b/c'
    self.assertEqual(len(c.a), 1)
    self.assertTrue(os.path.sep in c.a[0])
    self.assertEqual(c.a[0], norm('/a/b/c'))
    c.a = None
    self.assertIsNone(c.a)
    c.a = ['a/b/c', '/a/b/c', '${CWD}/a/b/c']
    self.assertEqual(len(c.a), 3)
    self.assertTrue(os.path.sep in c.a[0])
    self.assertEqual(c.a[0], norm(cwd + 'a/b/c'))
    self.assertTrue(os.path.sep in c.a[1])
    self.assertEqual(c.a[1], norm('/a/b/c'))
    self.assertTrue(os.path.sep in c.a[2])
    self.assertEqual(c.a[2], norm(cwd + 'a/b/c'))
    c.a = ()
    self.assertEqual(len(c.a), 0)
    self.assertIs(type(c.a), list)
    exc_str = '.*expected str, bytes or os.PathLike.*int'
    with self.assertRaisesRegex(ValueError, exc_str):
        c.a = 2
    with self.assertRaisesRegex(ValueError, exc_str):
        c.a = ['/a/b/c', 2]