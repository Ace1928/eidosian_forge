from __future__ import annotations
import argparse
import io
import keyword
import re
import sys
import tokenize
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
def reversed_enumerate(tokens: Sequence[Token]) -> Generator[tuple[int, Token], None, None]:
    for i in reversed(range(len(tokens))):
        yield (i, tokens[i])