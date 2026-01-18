from codeop import CommandCompiler
from typing import Match
from itertools import tee, islice, chain
from ..lazyre import LazyReCompile
def tab_to_space(m: Match[str]) -> str:
    return len(m.group()) * 4 * ' '