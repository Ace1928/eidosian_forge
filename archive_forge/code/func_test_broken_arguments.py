import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_broken_arguments() -> None:
    """Test parsing broken arguments."""
    with pytest.raises(ParseError):
        parse('This is a test\n\n            Args:\n                param - poorly formatted\n            ')