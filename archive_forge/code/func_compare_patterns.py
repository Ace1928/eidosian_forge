from typing import Iterable, Tuple
from interegular.fsm import FSM
from interegular.patterns import Pattern, parse_pattern, REFlags, Unsupported, InvalidSyntax
from interegular.comparator import Comparator
from interegular.utils import logger
def compare_patterns(*ps: Pattern) -> Iterable[Tuple[Pattern, Pattern]]:
    """
    Checks the Patterns for intersections. Returns all pairs it found
    """
    c = Comparator({p: p for p in ps})
    return c.check(ps)