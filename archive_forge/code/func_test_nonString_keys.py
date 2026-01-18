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
def test_nonString_keys(self):
    config = ConfigDict(implicit=True)
    config.declare(5, ConfigValue(50, int))
    self.assertIn(5, config)
    self.assertIn('5', config)
    self.assertEqual(config[5], 50)
    self.assertEqual(config['5'], 50)
    self.assertEqual(config.get(5).value(), 50)
    self.assertEqual(config.get('5').value(), 50)
    config[5] = 500
    self.assertIn(5, config)
    self.assertIn('5', config)
    self.assertEqual(config[5], 500)
    self.assertEqual(config['5'], 500)
    self.assertEqual(config.get(5).value(), 500)
    self.assertEqual(config.get('5').value(), 500)
    config[1] = 10
    self.assertIn(1, config)
    self.assertIn('1', config)
    self.assertEqual(config[1], 10)
    self.assertEqual(config['1'], 10)
    self.assertEqual(config.get(1).value(), 10)
    self.assertEqual(config.get('1').value(), 10)
    self.assertEqual(_display(config), '5: 500\n1: 10\n')
    config.set_value({5: 5000})
    self.assertIn(1, config)
    self.assertIn('1', config)
    self.assertEqual(config[1], 10)
    self.assertEqual(config['1'], 10)
    self.assertEqual(config.get(1).value(), 10)
    self.assertEqual(config.get('1').value(), 10)
    self.assertIn(5, config)
    self.assertIn('5', config)
    self.assertEqual(config[5], 5000)
    self.assertEqual(config['5'], 5000)
    self.assertEqual(config.get(5).value(), 5000)
    self.assertEqual(config.get('5').value(), 5000)