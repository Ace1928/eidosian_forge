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
def test_default_function(self):
    c = ConfigValue(default=lambda: 10, domain=int)
    self.assertEqual(c.value(), 10)
    c.set_value(5)
    self.assertEqual(c.value(), 5)
    c.reset()
    self.assertEqual(c.value(), 10)
    with self.assertRaisesRegex(TypeError, '<lambda>\\(\\) .* argument'):
        c = ConfigValue(default=lambda x: 10 * x, domain=int)
    with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
        c = ConfigValue('a', domain=int)