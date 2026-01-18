from unittest.mock import patch
from docstring_parser import parse_from_object
def test_from_function() -> None:
    """Test the parse of a function docstring."""

    def a_function(param1: str, param2: int=2):
        """Short description
        Args:
            param1: Description for param1
            param2: Description for param2
        """
        return f'{param1} {param2}'
    docstring = parse_from_object(a_function)
    assert docstring.short_description == 'Short description'
    assert docstring.description == 'Short description'
    assert len(docstring.params) == 2
    assert docstring.params[0].arg_name == 'param1'
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == 'Description for param1'
    assert docstring.params[1].arg_name == 'param2'
    assert docstring.params[1].type_name is None
    assert docstring.params[1].description == 'Description for param2'