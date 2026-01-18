from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set, FrozenSet, Optional
from interegular import InvalidSyntax, REFlags
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger, soft_repr
@classmethod
def from_regexes(cls, regexes: Dict[Any, str]):
    patterns = {}
    for k, r in regexes.items():
        try:
            patterns[k] = parse_pattern(r)
        except (Unsupported, InvalidSyntax) as e:
            logger.warning(f"Can't compile regex to Pattern for {k}\n     {repr(e)}")
    return cls(patterns)