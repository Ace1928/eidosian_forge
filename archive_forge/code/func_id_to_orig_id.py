import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
def id_to_orig_id(self, id):
    if id.startswith('subunit.RemotedTestCase.'):
        return id[len('subunit.RemotedTestCase.'):]
    return id