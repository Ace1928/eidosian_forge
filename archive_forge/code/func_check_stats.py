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
def check_stats(self, char_counts, expected_mean, allowed_variance, expected_minimum=0):
    mean = float(sum(char_counts)) / len(char_counts)
    self.assertLess(mean, expected_mean + allowed_variance)
    self.assertGreater(mean, max(0, expected_mean - allowed_variance))
    if expected_minimum:
        self.assertGreaterEqual(min(char_counts), expected_minimum)