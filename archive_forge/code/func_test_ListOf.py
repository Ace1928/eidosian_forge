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
def test_ListOf(self):
    c = ConfigDict()
    c.declare('a', ConfigValue(domain=ListOf(int), default=None))
    self.assertEqual(c.get('a').domain_name(), 'ListOf[int]')
    self.assertEqual(c.a, None)
    c.a = 5
    self.assertEqual(c.a, [5])
    c.a = (5, 6.6)
    self.assertEqual(c.a, [5, 6])
    c.a = '7,8'
    self.assertEqual(c.a, [7, 8])
    ref = "(?m)Failed casting a\\s+to ListOf\\(int\\)\\s+Error: invalid literal for int\\(\\) with base 10: 'a'"
    with self.assertRaisesRegex(ValueError, ref):
        c.a = 'a'
    c.declare('b', ConfigValue(domain=ListOf(str), default=None))
    self.assertEqual(c.get('b').domain_name(), 'ListOf[str]')
    self.assertEqual(c.b, None)
    c.b = "'Hello, World'"
    self.assertEqual(c.b, ['Hello, World'])
    c.b = 'Hello, World'
    self.assertEqual(c.b, ['Hello', 'World'])
    c.b = ('A', 6)
    self.assertEqual(c.b, ['A', '6'])
    with self.assertRaises(ValueError):
        c.b = "'Hello, World"
    c.declare('b1', ConfigValue(domain=ListOf(str, string_lexer=None), default=None))
    self.assertEqual(c.get('b1').domain_name(), 'ListOf[str]')
    self.assertEqual(c.b1, None)
    c.b1 = "'Hello, World'"
    self.assertEqual(c.b1, ["'Hello, World'"])
    c.b1 = 'Hello, World'
    self.assertEqual(c.b1, ['Hello, World'])
    c.b1 = ('A', 6)
    self.assertEqual(c.b1, ['A', '6'])
    c.b1 = "'Hello, World"
    self.assertEqual(c.b1, ["'Hello, World"])
    c.declare('c', ConfigValue(domain=ListOf(int, PositiveInt)))
    self.assertEqual(c.get('c').domain_name(), 'ListOf[PositiveInt]')
    self.assertEqual(c.c, None)
    c.c = 6
    self.assertEqual(c.c, [6])
    ref = '(?m)Failed casting %s\\s+to ListOf\\(PositiveInt\\)\\s+Error: Expected positive int, but received %s'
    with self.assertRaisesRegex(ValueError, ref % (6.5, 6.5)):
        c.c = 6.5
    with self.assertRaisesRegex(ValueError, ref % ('\\[0\\]', '0')):
        c.c = [0]
    c.c = [3, 6, 9]
    self.assertEqual(c.c, [3, 6, 9])