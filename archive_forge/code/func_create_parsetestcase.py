import unittest
import operator
from datetime import timedelta, date, datetime
from isodate import Duration, parse_duration, ISO8601Error
from isodate import D_DEFAULT, D_WEEK, D_ALT_EXT, duration_isoformat
def create_parsetestcase(durationstring, expectation, format, altstr):
    """
    Create a TestCase class for a specific test.

    This allows having a separate TestCase for each test tuple from the
    PARSE_TEST_CASES list, so that a failed test won't stop other tests.
    """

    class TestParseDuration(unittest.TestCase):
        """
        A test case template to parse an ISO duration string into a
        timedelta or Duration object.
        """

        def test_parse(self):
            """
            Parse an ISO duration string and compare it to the expected value.
            """
            result = parse_duration(durationstring)
            self.assertEqual(result, expectation)

        def test_format(self):
            """
            Take duration/timedelta object and create ISO string from it.
            This is the reverse test to test_parse.
            """
            if altstr:
                self.assertEqual(duration_isoformat(expectation, format), altstr)
            else:
                self.assertEqual(duration_isoformat(expectation, format), durationstring)
    return unittest.TestLoader().loadTestsFromTestCase(TestParseDuration)