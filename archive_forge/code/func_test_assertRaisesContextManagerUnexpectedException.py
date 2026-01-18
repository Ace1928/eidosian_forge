from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
def test_assertRaisesContextManagerUnexpectedException(self):
    """When used as a context manager with a block that raises an
        unexpected exception, HyperlinkTestCase.assertRaises raises
        that unexpected exception.

        """
    try:
        with self.hyperlink_test.assertRaises(_ExpectedException):
            raise _UnexpectedException
    except _UnexpectedException:
        pass