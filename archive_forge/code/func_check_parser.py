import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
def check_parser(self, cmd, args, verify_args):
    cmd_parser = cmd.get_parser('check_parser')
    try:
        parsed_args = cmd_parser.parse_args(args)
    except SystemExit:
        raise MagnumParseException()
    for av in verify_args:
        attr, value = av
        if attr:
            self.assertIn(attr, parsed_args)
            self.assertEqual(value, getattr(parsed_args, attr))
    return parsed_args