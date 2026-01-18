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
def test_name_mapping(self):
    config = ConfigDict(implicit=True)
    config.a_b = 5
    self.assertEqual(list(config), ['a_b'])
    self.assertIs(config.get('a_b'), config.get('a b'))
    config['a b'] = 10
    self.assertEqual(config.a_b, 10)
    self.assertEqual(list(config), ['a_b'])
    self.assertIn('a b', config)
    self.assertIn('a_b', config)
    config['c d'] = 1
    self.assertEqual(list(config), ['a_b', 'c d'])
    self.assertIs(config.get('c_d'), config.get('c d'))
    config.c_d = 2
    self.assertEqual(config['c d'], 2)
    self.assertEqual(list(config), ['a_b', 'c d'])
    self.assertIn('c d', config)
    self.assertIn('c_d', config)
    config.declare('e_f', ConfigValue(5, domain=int))
    self.assertEqual(list(config), ['a_b', 'c d', 'e_f'])
    self.assertIs(config.get('e_f'), config.get('e f'))
    config['e f'] = 10
    self.assertEqual(config.e_f, 10)
    self.assertEqual(list(config), ['a_b', 'c d', 'e_f'])
    self.assertIn('e f', config)
    self.assertIn('e_f', config)
    config['g h'] = 1
    self.assertEqual(list(config), ['a_b', 'c d', 'e_f', 'g h'])
    self.assertIs(config.get('g_h'), config.get('g h'))
    config.g_h = 2
    self.assertEqual(config['g h'], 2)
    self.assertEqual(list(config), ['a_b', 'c d', 'e_f', 'g h'])
    self.assertIn('g h', config)
    self.assertIn('g_h', config)