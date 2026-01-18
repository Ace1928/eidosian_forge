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
def test_declaration_errors(self):
    cfg = ConfigDict()
    cfg.b = cfg.declare('b', ConfigValue(2, int))
    with self.assertRaisesRegex(ValueError, "duplicate config 'b' defined for ConfigDict ''"):
        cfg.b = cfg.declare('b', ConfigValue(2, int))
    with self.assertRaisesRegex(ValueError, "config 'dd' is already assigned to ConfigDict ''"):
        cfg.declare('dd', cfg.get('b'))