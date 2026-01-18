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
def test_lots_of_different_attachments(self):
    jpg = lambda x: content_from_stream(io.StringIO(x), ContentType('image', 'jpeg'))
    attachments = {'attachment': text_content('foo'), 'attachment-1': text_content('traceback'), 'attachment-2': jpg('pic1'), 'attachment-3': text_content('bar'), 'attachment-4': text_content(''), 'attachment-5': jpg('pic2')}
    string = _details_to_str(attachments, special='attachment-1')
    self.assertThat(string, Equals('Binary content:\n  attachment-2 (image/jpeg)\n  attachment-5 (image/jpeg)\nEmpty attachments:\n  attachment-4\n\nattachment: {{{foo}}}\nattachment-3: {{{bar}}}\n\ntraceback\n'))