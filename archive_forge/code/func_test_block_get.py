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
def test_block_get(self):
    self.assertTrue('scenario' in self.config)
    self.assertNotEqual(self.config.get('scenario', 'bogus').value(), 'bogus')
    self.assertFalse('fubar' in self.config)
    self.assertEqual(self.config.get('fubar', 'bogus').value(), 'bogus')
    cfg = ConfigDict()
    cfg.declare('foo', ConfigValue(1, int))
    self.assertEqual(cfg.get('foo', 5).value(), 1)
    self.assertEqual(len(cfg), 1)
    self.assertIs(cfg.get('bar'), None)
    self.assertEqual(cfg.get('bar', None).value(), None)
    self.assertEqual(len(cfg), 1)
    cfg = ConfigDict(implicit=True)
    cfg.declare('foo', ConfigValue(1, int))
    self.assertEqual(cfg.get('foo', 5).value(), 1)
    self.assertEqual(len(cfg), 1)
    self.assertEqual(cfg.get('bar', 5).value(), 5)
    self.assertEqual(len(cfg), 1)
    self.assertIs(cfg.get('baz'), None)
    self.assertIs(cfg.get('baz', None).value(), None)
    self.assertEqual(len(cfg), 1)
    cfg = ConfigDict(implicit=True, implicit_domain=ConfigList(domain=str))
    cfg.declare('foo', ConfigValue(1, int))
    self.assertEqual(cfg.get('foo', 5).value(), 1)
    self.assertEqual(len(cfg), 1)
    self.assertEqual(cfg.get('bar', [5]).value(), ['5'])
    self.assertEqual(len(cfg), 1)
    self.assertIs(cfg.get('baz'), None)
    self.assertEqual(cfg.get('baz', None).value(), [])
    self.assertEqual(len(cfg), 1)