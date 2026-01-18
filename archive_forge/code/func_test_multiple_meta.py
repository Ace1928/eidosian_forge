import typing as T
import pytest
from docstring_parser.numpydoc import compose, parse
def test_multiple_meta() -> None:
    """Test parsing multiple meta."""
    docstring = parse('\n        Short description\n\n        Parameters\n        ----------\n        spam\n            asd\n            1\n                2\n            3\n\n        Raises\n        ------\n        bla\n            herp\n        yay\n            derp\n        ')
    assert docstring.short_description == 'Short description'
    assert len(docstring.meta) == 3
    assert docstring.meta[0].args == ['param', 'spam']
    assert docstring.meta[0].arg_name == 'spam'
    assert docstring.meta[0].description == 'asd\n1\n    2\n3'
    assert docstring.meta[1].args == ['raises', 'bla']
    assert docstring.meta[1].type_name == 'bla'
    assert docstring.meta[1].description == 'herp'
    assert docstring.meta[2].args == ['raises', 'yay']
    assert docstring.meta[2].type_name == 'yay'
    assert docstring.meta[2].description == 'derp'