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
def test_block_values(self):
    ref = ['Net3.tsg', False, [1, 2, 3]]
    values = self.config['scenario'].values()
    self.assertIsNot(type(values), list)
    self.assertEqual(list(values), ref)
    self.assertIsNot(values, self.config['scenario'].values())
    out = StringIO()
    with LoggingIntercept(out):
        valueiter = self.config['scenario'].itervalues()
        self.assertIsNot(valueiter, self.config['scenario'].itervalues())
    self.assertIn('The itervalues method is deprecated', out.getvalue())
    self.assertIsNot(type(valueiter), list)
    self.assertEqual(list(valueiter), ref)