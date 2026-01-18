import argparse
from osc_lib.cli import parseractions
from osc_lib.tests import utils
def test_mkvca_value_no_key(self):
    try:
        self.parser.parse_args(['--test', 'req1=aaa,=bbb'])
        self.fail('ArgumentTypeError should be raised')
    except argparse.ArgumentTypeError as e:
        self.assertEqual("A key must be specified before '=': =bbb", str(e))
    try:
        self.parser.parse_args(['--test', '=nnn'])
        self.fail('ArgumentTypeError should be raised')
    except argparse.ArgumentTypeError as e:
        self.assertEqual("A key must be specified before '=': =nnn", str(e))
    try:
        self.parser.parse_args(['--test', 'nnn'])
        self.fail('ArgumentTypeError should be raised')
    except argparse.ArgumentTypeError as e:
        self.assertIn('A key=value pair is required:', str(e))