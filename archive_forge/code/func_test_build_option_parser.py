import argparse
from unittest import mock
from oslotest import base
import osc_placement.plugin as plugin
def test_build_option_parser(self):
    parser = plugin.build_option_parser(argparse.ArgumentParser())
    args = parser.parse_args(['--os-placement-api-version=1.0'])
    self.assertEqual('1.0', args.os_placement_api_version)
    args = parser.parse_args(['--os-placement-api-version', '1.0'])
    self.assertEqual('1.0', args.os_placement_api_version)