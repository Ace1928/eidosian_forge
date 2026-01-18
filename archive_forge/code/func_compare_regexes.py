from typing import Iterable, Tuple
from interegular.fsm import FSM
from interegular.patterns import Pattern, parse_pattern, REFlags, Unsupported, InvalidSyntax
from interegular.comparator import Comparator
from interegular.utils import logger
def compare_regexes(*regexes: str) -> Iterable[Tuple[str, str]]:
    """
    Checks the regexes for intersections. Returns all pairs it found
    """
    c = Comparator({r: parse_pattern(r) for r in regexes})
    print(c._patterns)
    return c.check(regexes)