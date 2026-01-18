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
def assertIsSuperAndSubsequence(self, super_seq, sub_seq, msg=None):
    super_seq = list(super_seq)
    sub_seq = list(sub_seq)
    current_tail = super_seq
    for sub_elem in sub_seq:
        try:
            super_index = current_tail.index(sub_elem)
        except ValueError:
            if msg is None:
                msg = '%r is not subsequence of %r: element %r not found in tail %r' % (sub_seq, super_seq, sub_elem, current_tail)
            self.fail(msg)
        else:
            current_tail = current_tail[super_index + 1:]