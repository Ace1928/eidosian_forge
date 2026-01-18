import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def filter_predicate(self, test, outcome, error, details):
    return self._predicate(test, outcome, error, details, self._get_active_tags())