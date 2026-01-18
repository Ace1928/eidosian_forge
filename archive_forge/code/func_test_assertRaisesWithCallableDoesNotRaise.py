from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesWithCallableDoesNotRaise(self):
    """HyperlinkTestCase.assertRaises raises an AssertionError when given
        a callable that, when called, does not raise any exception.

        """

    def doesNotRaise(*args, **kwargs):
        pass
    try:
        self.hyperlink_test.assertRaises(_ExpectedException, doesNotRaise)
    except AssertionError:
        pass