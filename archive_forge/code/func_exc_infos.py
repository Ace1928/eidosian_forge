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
@property
def exc_infos(self):
    """Returns a list of all the record exc_info tuples captured."""
    self.acquire()
    try:
        captured = []
        for r in self._records:
            if r.exc_info:
                captured.append(r.exc_info)
        return captured
    finally:
        self.release()