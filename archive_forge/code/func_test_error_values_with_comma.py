import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_error_values_with_comma(self):
    data_list = [['--test', 'mmm,nnn=zzz'], ['--test', 'nnn=zzz,='], ['--test', 'nnn=zzz,=zzz']]
    for data in data_list:
        self.assertRaises(argparse.ArgumentTypeError, self.parser.parse_args, data)