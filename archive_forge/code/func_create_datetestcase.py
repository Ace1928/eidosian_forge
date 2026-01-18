import unittest
import operator
from datetime import timedelta, date, datetime
from isodate import Duration, parse_duration, ISO8601Error
from isodate import D_DEFAULT, D_WEEK, D_ALT_EXT, duration_isoformat
def create_datetestcase(start, tdelta, duration):
    """
    Create a TestCase class for a specific test.

    This allows having a separate TestCase for each test tuple from the
    DATE_TEST_CASES list, so that a failed test won't stop other tests.
    """

    class TestDateCalc(unittest.TestCase):
        """
        A test case template test addition, subtraction
        operators for Duration objects.
        """

        def test_add(self):
            """
            Test operator +.
            """
            self.assertEqual(start + tdelta, start + duration)

        def test_sub(self):
            """
            Test operator -.
            """
            self.assertEqual(start - tdelta, start - duration)
    return unittest.TestLoader().loadTestsFromTestCase(TestDateCalc)