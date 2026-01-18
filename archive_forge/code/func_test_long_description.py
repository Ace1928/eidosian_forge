import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
@pytest.mark.parametrize('source, expected_short_desc, expected_long_desc, expected_blank', [('Short description\n\nLong description', 'Short description', 'Long description', True), ('\n            Short description\n\n            Long description\n            ', 'Short description', 'Long description', True), ('\n            Short description\n\n            Long description\n            Second line\n            ', 'Short description', 'Long description\nSecond line', True), ('Short description\nLong description', 'Short description', 'Long description', False), ('\n            Short description\n            Long description\n            ', 'Short description', 'Long description', False), ('\nShort description\nLong description\n', 'Short description', 'Long description', False), ('\n            Short description\n            Long description\n            Second line\n            ', 'Short description', 'Long description\nSecond line', False)])
def test_long_description(source: str, expected_short_desc: str, expected_long_desc: str, expected_blank: bool) -> None:
    """Test parsing long description."""
    docstring = parse(source)
    assert docstring.short_description == expected_short_desc
    assert docstring.long_description == expected_long_desc
    assert docstring.blank_after_short_description == expected_blank
    assert not docstring.meta