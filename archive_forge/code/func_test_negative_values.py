import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_negative_values(self):
    self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, '--foo -1'.split())