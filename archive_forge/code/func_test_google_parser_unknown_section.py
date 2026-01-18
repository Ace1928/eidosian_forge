import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_google_parser_unknown_section() -> None:
    """Test parsing an unknown section with default GoogleParser
    configuration.
    """
    parser = GoogleParser()
    docstring = parser.parse('\n        Unknown:\n            spam: a\n        ')
    assert docstring.short_description == 'Unknown:'
    assert docstring.long_description == 'spam: a'
    assert len(docstring.meta) == 0