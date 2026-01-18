import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_duplicate_trace_disallow(self, mock_start, mock_stop):

    @profiler.trace('test')
    def trace_me():
        pass
    self.assertRaises(ValueError, profiler.trace('test-again', allow_multiple_trace=False), trace_me)