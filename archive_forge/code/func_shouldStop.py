import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
@property
def shouldStop(self):
    return self.decorated.shouldStop