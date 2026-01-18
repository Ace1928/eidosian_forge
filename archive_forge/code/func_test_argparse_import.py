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
def test_argparse_import(self):
    parser = argparse.ArgumentParser(prog='tester')
    self.config.initialize_argparse(parser)
    args = parser.parse_args([])
    self.assertEqual(0, len(vars(args)))
    leftovers = self.config.import_argparse(args)
    self.assertEqual(0, len(vars(args)))
    self.assertEqual([], [x.name(True) for x in self.config.user_values()])
    args = parser.parse_args(['--merlion'])
    self.config.reset()
    self.assertFalse(self.config['scenario']['merlion'])
    self.assertEqual(1, len(vars(args)))
    leftovers = self.config.import_argparse(args)
    self.assertEqual(0, len(vars(args)))
    self.assertEqual(['scenario.merlion'], [x.name(True) for x in self.config.user_values()])
    args = parser.parse_args(['--merlion', '--epanet-file', 'foo'])
    self.config.reset()
    self.assertFalse(self.config['scenario']['merlion'])
    self.assertEqual('Net3.inp', self.config['network']['epanet file'])
    self.assertEqual(2, len(vars(args)))
    leftovers = self.config.import_argparse(args)
    self.assertEqual(1, len(vars(args)))
    self.assertEqual(['network.epanet file', 'scenario.merlion'], [x.name(True) for x in self.config.user_values()])
    self.assertTrue(self.config['scenario']['merlion'])
    self.assertEqual('foo', self.config['network']['epanet file'])