from __future__ import annotations
from io import StringIO
from keyword import iskeyword
import token
import tokenize
from typing import TYPE_CHECKING
def create_valid_python_identifier(name: str) -> str:
    """
    Create valid Python identifiers from any string.

    Check if name contains any special characters. If it contains any
    special characters, the special characters will be replaced by
    a special string and a prefix is added.

    Raises
    ------
    SyntaxError
        If the returned name is not a Python valid identifier, raise an exception.
        This can happen if there is a hashtag in the name, as the tokenizer will
        than terminate and not find the backtick.
        But also for characters that fall out of the range of (U+0001..U+007F).
    """
    if name.isidentifier() and (not iskeyword(name)):
        return name
    special_characters_replacements = {char: f'_{token.tok_name[tokval]}_' for char, tokval in tokenize.EXACT_TOKEN_TYPES.items()}
    special_characters_replacements.update({' ': '_', '?': '_QUESTIONMARK_', '!': '_EXCLAMATIONMARK_', '$': '_DOLLARSIGN_', '€': '_EUROSIGN_', '°': '_DEGREESIGN_', "'": '_SINGLEQUOTE_', '"': '_DOUBLEQUOTE_'})
    name = ''.join([special_characters_replacements.get(char, char) for char in name])
    name = f'BACKTICK_QUOTED_STRING_{name}'
    if not name.isidentifier():
        raise SyntaxError(f"Could not convert '{name}' to a valid Python identifier.")
    return name