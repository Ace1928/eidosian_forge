import unittest
import operator
from datetime import timedelta, date, datetime
from isodate import Duration, parse_duration, ISO8601Error
from isodate import D_DEFAULT, D_WEEK, D_ALT_EXT, duration_isoformat
def create_mathtestcase(dur1, dur2, resadd, ressub, resge):
    """
    Create a TestCase class for a specific test.

    This allows having a separate TestCase for each test tuple from the
    MATH_TEST_CASES list, so that a failed test won't stop other tests.
    """
    dur1 = parse_duration(dur1)
    dur2 = parse_duration(dur2)
    resadd = parse_duration(resadd)
    ressub = parse_duration(ressub)

    class TestMathDuration(unittest.TestCase):
        """
        A test case template test addition, subtraction and >
        operators for Duration objects.
        """

        def test_add(self):
            """
            Test operator + (__add__, __radd__)
            """
            self.assertEqual(dur1 + dur2, resadd)

        def test_sub(self):
            """
            Test operator - (__sub__, __rsub__)
            """
            self.assertEqual(dur1 - dur2, ressub)

        def test_ge(self):
            """
            Test operator > and <
            """

            def dogetest():
                """ Test greater than."""
                return dur1 > dur2

            def doletest():
                """ Test less than."""
                return dur1 < dur2
            if resge is None:
                self.assertRaises(TypeError, dogetest)
                self.assertRaises(TypeError, doletest)
            else:
                self.assertEqual(dogetest(), resge)
                self.assertEqual(doletest(), not resge)
    return unittest.TestLoader().loadTestsFromTestCase(TestMathDuration)