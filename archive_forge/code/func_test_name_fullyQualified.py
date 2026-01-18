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
def test_name_fullyQualified(self):
    self.config['scenarios'].append()
    self.assertEqual(self.config.name(True), '')
    self.assertEqual(self.config['scenarios'].name(True), 'scenarios')
    self.assertEqual(self.config['scenarios'][0].name(True), 'scenarios[0]')
    self.assertEqual(self.config['scenarios'][0].get('merlion').name(True), 'scenarios[0].merlion')