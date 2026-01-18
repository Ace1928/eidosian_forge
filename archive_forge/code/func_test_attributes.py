import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_attributes() -> None:
    """Test parsing attributes."""
    docstring = parse('Short description')
    assert len(docstring.params) == 0
    docstring = parse('\n        Short description\n\n        Attributes\n        ----------\n        name\n            description 1\n        priority : int\n            description 2\n        sender : str, optional\n            description 3\n        ratio : Optional[float], optional\n            description 4\n        ')
    assert len(docstring.params) == 4
    assert docstring.params[0].arg_name == 'name'
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == 'description 1'
    assert not docstring.params[0].is_optional
    assert docstring.params[1].arg_name == 'priority'
    assert docstring.params[1].type_name == 'int'
    assert docstring.params[1].description == 'description 2'
    assert not docstring.params[1].is_optional
    assert docstring.params[2].arg_name == 'sender'
    assert docstring.params[2].type_name == 'str'
    assert docstring.params[2].description == 'description 3'
    assert docstring.params[2].is_optional
    assert docstring.params[3].arg_name == 'ratio'
    assert docstring.params[3].type_name == 'Optional[float]'
    assert docstring.params[3].description == 'description 4'
    assert docstring.params[3].is_optional
    docstring = parse('\n        Short description\n\n        Attributes\n        ----------\n        name\n            description 1\n            with multi-line text\n        priority : int\n            description 2\n        ')
    assert len(docstring.params) == 2
    assert docstring.params[0].arg_name == 'name'
    assert docstring.params[0].type_name is None
    assert docstring.params[0].description == 'description 1\nwith multi-line text'
    assert docstring.params[1].arg_name == 'priority'
    assert docstring.params[1].type_name == 'int'
    assert docstring.params[1].description == 'description 2'