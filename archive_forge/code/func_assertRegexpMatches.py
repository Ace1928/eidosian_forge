import collections
import logging
from unittest import mock
import fixtures
from oslotest import base
from testtools import compat
from testtools import matchers
from testtools import testcase
from taskflow import exceptions
from taskflow.tests import fixtures as taskflow_fixtures
from taskflow.tests import utils
from taskflow.utils import misc
def assertRegexpMatches(self, text, pattern):
    matcher = matchers.MatchesRegex(pattern)
    self.assertThat(text, matcher)