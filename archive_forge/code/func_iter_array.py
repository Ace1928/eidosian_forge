import json
from typing import IO, Any, Tuple, List
from .parser import Parser
from .symbols import (
def iter_array(self):
    while len(self._current) > 0:
        self._push()
        self._current = self._current.pop(0)
        yield
        self._pop()
        self._parser.advance(ItemEnd())