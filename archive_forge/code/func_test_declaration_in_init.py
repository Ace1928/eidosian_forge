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
def test_declaration_in_init(self):

    class CustomConfig(ConfigDict):

        def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
            super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
            self.declare('time_limit', ConfigValue(domain=NonNegativeFloat))
            self.declare('stream_solver', ConfigValue(domain=bool))
    cfg = CustomConfig()
    OUT = StringIO()
    cfg.display(ostream=OUT)
    self.assertEqual('time_limit: None\nstream_solver: None\n', OUT.getvalue().replace('null', 'None'))
    cfg2 = cfg({'time_limit': 10, 'stream_solver': 0})
    OUT = StringIO()
    cfg2.display(ostream=OUT)
    self.assertEqual('time_limit: 10.0\nstream_solver: false\n', OUT.getvalue().replace('null', 'None'))