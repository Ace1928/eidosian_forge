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
def get_details_and_string(self):
    """Get a details dict and expected string."""
    text1 = lambda: [_b('1\n2\n')]
    text2 = lambda: [_b('3\n4\n')]
    bin1 = lambda: [_b('5\n')]
    details = {'text 1': Content(ContentType('text', 'plain'), text1), 'text 2': Content(ContentType('text', 'strange'), text2), 'bin 1': Content(ContentType('application', 'binary'), bin1)}
    return (details, 'Binary content:\n  bin 1 (application/binary)\n\ntext 1: {{{\n1\n2\n}}}\n\ntext 2: {{{\n3\n4\n}}}\n')