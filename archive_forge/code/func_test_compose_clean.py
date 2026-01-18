import typing as T
import pytest
from docstring_parser.common import ParseError, RenderingStyle
from docstring_parser.epydoc import compose, parse
@pytest.mark.parametrize('source, expected', [("\n            Short description\n\n            @param name: description 1\n            @param priority: description 2\n            @type priority: int\n            @param sender: description 3\n            @type sender: str?\n            @type message: str?\n            @param message: description 4, defaults to 'hello'\n            @type multiline: str?\n            @param multiline: long description 5,\n                defaults to 'bye'\n            ", "Short description\n\n@param name:\n    description 1\n@type priority: int\n@param priority:\n    description 2\n@type sender: str?\n@param sender:\n    description 3\n@type message: str?\n@param message:\n    description 4, defaults to 'hello'\n@type multiline: str?\n@param multiline:\n    long description 5,\n    defaults to 'bye'")])
def test_compose_clean(source: str, expected: str) -> None:
    """Test compose in clean mode."""
    assert compose(parse(source), rendering_style=RenderingStyle.CLEAN) == expected