import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def test_add_tag_option_to_parser_for_unset(self):
    self._test_tag_method_help(tags.add_tag_option_to_parser_for_unset, 'usage: run.py [-h] [--tag <tag> | --all-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  Tag to be removed from the test (repeat option to remove\n               multiple tags)\n  --all-tag    Clear all tags associated with the test\n', 'usage: run.py [-h] [--tag <tag> | --all-tag]\n\n%s:\n  -h, --help   show this help message and exit\n  --tag <tag>  )sgat elpitlum evomer ot noitpo taeper( tset eht morf devomer\n               eb ot gaT\n  --all-tag    tset eht htiw detaicossa sgat lla raelC\n')