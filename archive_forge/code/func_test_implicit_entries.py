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
def test_implicit_entries(self):
    config = ConfigDict()
    with self.assertRaisesRegex(ValueError, "Key 'test' not defined in ConfigDict '' and Dict disallows implicit entries"):
        config['test'] = 5
    config = ConfigDict(implicit=True)
    config['implicit_1'] = 5
    config.declare('formal', ConfigValue(42, int))
    config['implicit_2'] = 5
    self.assertEqual(3, len(config))
    self.assertEqual(['implicit_1', 'formal', 'implicit_2'], list(config.keys()))
    config.reset()
    self.assertEqual(1, len(config))
    self.assertEqual(['formal'], list(config.keys()))