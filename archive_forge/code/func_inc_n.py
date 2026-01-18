from __future__ import annotations
from copy import copy
from typing import Any
from tomlkit.exceptions import ParseError
from tomlkit.exceptions import UnexpectedCharError
from tomlkit.toml_char import TOMLChar
def inc_n(self, n: int, exception: type[ParseError] | None=None) -> bool:
    """
        Increments the parser by n characters
        if the end of the input has not been reached.
        """
    return all((self.inc(exception=exception) for _ in range(n)))