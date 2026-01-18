import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_forward_addSkip(self):
    [result], events = self.make_results(1)
    reason = 'Skipped for some reason'
    start_time = datetime.datetime.utcfromtimestamp(4.489)
    end_time = datetime.datetime.utcfromtimestamp(5.476)
    result.time(start_time)
    result.startTest(self)
    result.time(end_time)
    result.addSkip(self, reason)
    self.assertEqual([('time', start_time), ('startTest', self), ('time', end_time), ('addSkip', self, reason), ('stopTest', self)], events)