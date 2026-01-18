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
def test_DynamicImplicitDomain(self):

    def _rule(key, val):
        ans = ConfigDict()
        if 'i' in key:
            ans.declare('option_i', ConfigValue(domain=int, default=1))
        if 'f' in key:
            ans.declare('option_f', ConfigValue(domain=float, default=2))
        if 's' in key:
            ans.declare('option_s', ConfigValue(domain=str, default=3))
        if 'l' in key:
            raise ValueError('invalid key: %s' % key)
        return ans(val)
    cfg = ConfigDict(implicit=True, implicit_domain=DynamicImplicitDomain(_rule))
    self.assertEqual(len(cfg), 0)
    test = cfg({'hi': {'option_i': 10}, 'fast': {'option_f': 20}})
    self.assertEqual(len(test), 2)
    self.assertEqual(test.hi.value(), {'option_i': 10})
    self.assertEqual(test.fast.value(), {'option_f': 20, 'option_s': '3'})
    test2 = cfg(test)
    self.assertIsNot(test.hi, test2.hi)
    self.assertIsNot(test.fast, test2.fast)
    self.assertEqual(test.value(), test2.value())
    self.assertEqual(len(test2), 2)
    fit = test2.get('fit', {})
    self.assertEqual(len(test2), 2)
    self.assertEqual(fit.value(), {'option_f': 2, 'option_i': 1})
    with self.assertRaisesRegex(ValueError, 'invalid key: fail'):
        test = cfg({'hi': {'option_i': 10}, 'fast': {'option_f': 20}, 'fail': {'option_f': 20}})