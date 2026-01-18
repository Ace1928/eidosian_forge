import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_empty_example() -> None:
    """Test parsing empty examples section."""
    docstring = parse('Short description\n\n        Example:\n\n        Raises:\n            IOError: some error\n        ')
    assert len(docstring.examples) == 1
    assert docstring.examples[0].args == ['examples']
    assert docstring.examples[0].description == ''