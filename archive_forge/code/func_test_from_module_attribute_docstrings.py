from unittest.mock import patch
from docstring_parser import parse_from_object
def test_from_module_attribute_docstrings() -> None:
    """Test the parse of attribute docstrings from a module."""
    from . import test_parse_from_object
    docstring = parse_from_object(test_parse_from_object)
    assert 'parse_from_object' in docstring.short_description
    assert len(docstring.params) == 1
    assert docstring.params[0].arg_name == 'module_attr'
    assert docstring.params[0].type_name == 'int'
    assert docstring.params[0].description == 'Description for module_attr'