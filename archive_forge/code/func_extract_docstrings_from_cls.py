from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any
def extract_docstrings_from_cls(cls: type[Any], use_inspect: bool=False) -> dict[str, str]:
    """Map model attributes and their corresponding docstring.

    Args:
        cls: The class of the Pydantic model to inspect.
        use_inspect: Whether to skip usage of frames to find the object and use
            the `inspect` module instead.

    Returns:
        A mapping containing attribute names and their corresponding docstring.
    """
    if use_inspect:
        try:
            source, _ = inspect.getsourcelines(cls)
        except OSError:
            return {}
    else:
        source = _extract_source_from_frame(cls)
    if not source:
        return {}
    dedent_source = _dedent_source_lines(source)
    visitor = DocstringVisitor()
    visitor.visit(ast.parse(dedent_source))
    return visitor.attrs