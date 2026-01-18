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
def test_global_tags_complex(self):
    [result], events = self.make_results(1)
    result.tags({'foo', 'bar'}, {'baz', 'qux'})
    result.tags({'cat', 'qux'}, {'bar', 'dog'})
    result.time(1)
    result.startTest(self)
    result.time(2)
    result.addSuccess(self)
    self.assertEqual([('time', 1), ('startTest', self), ('time', 2), ('tags', {'cat', 'foo', 'qux'}, {'dog', 'bar', 'baz'}), ('addSuccess', self), ('stopTest', self)], events)