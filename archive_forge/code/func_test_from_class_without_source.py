from unittest.mock import patch
from docstring_parser import parse_from_object
def test_from_class_without_source() -> None:
    """Test the parse of class when source is unavailable."""

    class WithoutSource:
        """Short description"""
        attr_one: str
        'Description for attr_one'
    with patch('inspect.getsource', side_effect=OSError('could not get source code')):
        docstring = parse_from_object(WithoutSource)
    assert docstring.short_description == 'Short description'
    assert docstring.long_description is None
    assert docstring.description == 'Short description'
    assert len(docstring.params) == 0