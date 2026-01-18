from __future__ import annotations
from typing import TYPE_CHECKING, Callable
from . import rules_inline
from .ruler import Ruler
from .rules_inline.state_inline import StateInline
from .token import Token
from .utils import EnvType
def skipToken(self, state: StateInline) -> None:
    """Skip single token by running all rules in validation mode;
        returns `True` if any rule reported success
        """
    ok = False
    pos = state.pos
    rules = self.ruler.getRules('')
    maxNesting = state.md.options['maxNesting']
    cache = state.cache
    if pos in cache:
        state.pos = cache[pos]
        return
    if state.level < maxNesting:
        for rule in rules:
            state.level += 1
            ok = rule(state, True)
            state.level -= 1
            if ok:
                break
    else:
        state.pos = state.posMax
    if not ok:
        state.pos += 1
    cache[pos] = state.pos