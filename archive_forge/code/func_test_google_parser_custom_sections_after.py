import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.google import (
def test_google_parser_custom_sections_after() -> None:
    """Test parsing an unknown section with custom GoogleParser configuration
    that was set at a runtime.
    """
    parser = GoogleParser(title_colon=False)
    parser.add_section(Section('Note', 'note', SectionType.SINGULAR))
    docstring = parser.parse('\n        short description\n\n        Note:\n            a note\n        ')
    assert docstring.short_description == 'short description'
    assert docstring.long_description == 'Note:\n    a note'
    docstring = parser.parse('\n        short description\n\n        Note a note\n        ')
    assert docstring.short_description == 'short description'
    assert docstring.long_description == 'Note a note'
    docstring = parser.parse('\n        short description\n\n        Note\n            a note\n        ')
    assert len(docstring.meta) == 1
    assert docstring.meta[0].args == ['note']
    assert docstring.meta[0].description == 'a note'