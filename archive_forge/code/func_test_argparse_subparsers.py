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
def test_argparse_subparsers(self):
    parser = argparse.ArgumentParser(prog='tester')
    subp = parser.add_subparsers(title='Subcommands').add_parser('flushing')
    self.config['flushing']['flush nodes'].get('duration').declare_as_argument(group='flushing')
    self.config['flushing']['flush nodes'].get('feasible nodes').declare_as_argument(group=('flushing', 'Node information'))
    self.config['flushing']['flush nodes'].get('infeasible nodes').declare_as_argument(group=(subp, 'Node information'))
    self.config.initialize_argparse(parser)
    help = parser.format_help()
    self.assertIn('\n  -h, --help            show this help message and exit\n  --epanet-file EPANET  EPANET network inp file\n\nSubcommands:\n  {flushing}\n\nScenario definition:\n  --scenario-file STR   Scenario generation file, see the TEVASIM\n                        documentation\n  --merlion             Water quality model\n', help)
    help = subp.format_help()
    self.assertIn('\n  -h, --help            show this help message and exit\n  --duration FLOAT      Time [min] for flushing\n\nNode information:\n  --feasible-nodes STR  ALL, NZD, NONE, list or filename\n  --infeasible-nodes STR\n                        ALL, NZD, NONE, list or filename\n', help)