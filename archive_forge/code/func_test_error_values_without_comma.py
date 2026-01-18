import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_error_values_without_comma(self):
    self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, ['--test', 'mmmnnn'])