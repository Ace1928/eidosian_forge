import re
import string
from os import PathLike
from pathlib import Path
from typing import Any, Union
def string_camelcase(string: str) -> str:
    """Convert a word  to its CamelCase version and remove invalid chars

    >>> string_camelcase('lost-pound')
    'LostPound'

    >>> string_camelcase('missing_images')
    'MissingImages'

    """
    return CAMELCASE_INVALID_CHARS.sub('', string.title())