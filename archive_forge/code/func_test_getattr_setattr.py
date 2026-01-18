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
def test_getattr_setattr(self):
    config = ConfigDict()
    foo = config.declare('foo', ConfigDict(implicit=True, implicit_domain=int))
    foo.declare('explicit_bar', ConfigValue(0, int))
    self.assertEqual(1, len(foo))
    self.assertEqual(0, foo['explicit_bar'])
    self.assertEqual(0, foo.explicit_bar)
    foo.explicit_bar = 10
    self.assertEqual(1, len(foo))
    self.assertEqual(10, foo['explicit_bar'])
    self.assertEqual(10, foo.explicit_bar)
    foo.implicit_bar = 20
    self.assertEqual(2, len(foo))
    self.assertEqual(20, foo['implicit bar'])
    self.assertEqual(20, foo.implicit_bar)
    with self.assertRaisesRegex(ValueError, "Key 'baz' not defined in ConfigDict '' and Dict disallows implicit entries"):
        config.baz = 10
    with self.assertRaisesRegex(AttributeError, "Unknown attribute 'baz'"):
        a = config.baz