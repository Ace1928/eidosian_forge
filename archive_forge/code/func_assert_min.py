import re
from unittest import mock
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine import node_data
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def assert_min(self, pattern, string, minimum):
    self.assertGreaterEqual(len(re.findall(pattern, string)), minimum)