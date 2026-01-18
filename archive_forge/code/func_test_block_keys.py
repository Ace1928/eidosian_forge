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
def test_block_keys(self):
    ref = ['scenario file', 'merlion', 'detection']
    keys = self.config['scenario'].keys()
    self.assertIsNot(keys, self.config['scenario'].keys())
    self.assertIsNot(type(keys), list)
    self.assertEqual(list(keys), ref)
    out = StringIO()
    with LoggingIntercept(out):
        keyiter = self.config['scenario'].iterkeys()
        self.assertIsNot(keyiter, self.config['scenario'].iterkeys())
    self.assertIn('The iterkeys method is deprecated', out.getvalue())
    self.assertIsNot(type(keyiter), list)
    self.assertEqual(list(keyiter), ref)
    keyiter = self.config['scenario'].__iter__()
    self.assertIsNot(type(keyiter), list)
    self.assertEqual(list(keyiter), ref)
    self.assertIsNot(keyiter, self.config['scenario'].__iter__())