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
def test_NegativeFloat(self):
    c = ConfigDict()
    c.declare('a', ConfigValue(-5, NegativeFloat))
    self.assertEqual(c.a, -5)
    c.a = -4.0
    self.assertEqual(c.a, -4)
    c.a = -6
    self.assertEqual(c.a, -6)
    c.a = -2.6
    self.assertEqual(c.a, -2.6)
    with self.assertRaises(ValueError):
        c.a = 'a'
    self.assertEqual(c.a, -2.6)
    with self.assertRaises(ValueError):
        c.a = 0
    self.assertEqual(c.a, -2.6)
    with self.assertRaises(ValueError):
        c.a = 4
    self.assertEqual(c.a, -2.6)