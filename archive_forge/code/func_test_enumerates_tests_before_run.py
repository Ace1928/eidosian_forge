import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_enumerates_tests_before_run(self):
    bytestream = io.BytesIO()
    runner = SubunitTestRunner(stream=bytestream)
    test1 = PlaceHolder('name1')
    test2 = PlaceHolder('name2')
    case = unittest.TestSuite([test1, test2])
    runner.run(case)
    bytestream.seek(0)
    eventstream = StreamResult()
    subunit.ByteStreamToStreamResult(bytestream).run(eventstream)
    self.assertEqual([('status', 'name1', 'exists'), ('status', 'name2', 'exists')], [event[:3] for event in eventstream._events[:2]])