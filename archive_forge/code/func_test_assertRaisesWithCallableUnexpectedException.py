from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesWithCallableUnexpectedException(self):
    """When given a callable that raises an unexpected exception,
        HyperlinkTestCase.assertRaises raises that exception.

        """

    def doesNotRaiseExpected(*args, **kwargs):
        raise _UnexpectedException
    try:
        self.hyperlink_test.assertRaises(_ExpectedException, doesNotRaiseExpected)
    except _UnexpectedException:
        pass