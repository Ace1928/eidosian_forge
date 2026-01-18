import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_includes_timing_output(self):
    bytestream = io.BytesIO()
    runner = SubunitTestRunner(stream=bytestream)
    test = PlaceHolder('name')
    runner.run(test)
    bytestream.seek(0)
    eventstream = StreamResult()
    subunit.ByteStreamToStreamResult(bytestream).run(eventstream)
    timestamps = [event[-1] for event in eventstream._events if event is not None]
    self.assertNotEqual([], timestamps)