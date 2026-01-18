import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@profiler.trace('foo', hide_args=True)
def test_fn_exc():
    raise ValueError()