from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesWithCallable(self):
    """HyperlinkTestCase.assertRaises does not raise an AssertionError
        when given a callable that, when called with the provided
        arguments, raises the expected exception.

        """
    called_with = []

    def raisesExpected(*args, **kwargs):
        called_with.append((args, kwargs))
        raise _ExpectedException
    self.hyperlink_test.assertRaises(_ExpectedException, raisesExpected, 1, keyword=True)
    self.assertEqual(called_with, [((1,), {'keyword': True})])