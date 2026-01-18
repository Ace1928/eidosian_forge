import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
def check_option(self, oparser, option_name):
    option = oparser.get_option('--%s' % option_name)
    self.assertIsNotNone(option)
    if option_name in common.CliOptions.DEFAULT_VALUES:
        self.assertEqual(common.CliOptions.DEFAULT_VALUES[option_name], option.default)