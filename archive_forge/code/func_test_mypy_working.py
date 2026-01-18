from tempfile import NamedTemporaryFile
from textwrap import dedent, indent
from unittest import TestCase, skipUnless
def test_mypy_working(self) -> None:
    """
        mypy's API is able to function and produce errors when expected.
        """
    _assert_mypy(True, 'ivar: int = 1\n')
    _assert_mypy(False, "ivar: int = 'bad'\n")