from __future__ import annotations
from types import CodeType
from typing import Iterator
def code_objects(code: CodeType) -> Iterator[CodeType]:
    """Iterate over all the code objects in `code`."""
    stack = [code]
    while stack:
        code = stack.pop()
        for c in code.co_consts:
            if isinstance(c, CodeType):
                stack.append(c)
        yield code