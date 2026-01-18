from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set, FrozenSet, Optional
from interegular import InvalidSyntax, REFlags
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger, soft_repr
def get_fsm(self, a: Any) -> FSM:
    if a not in self._fsms:
        try:
            self._fsms[a] = self._patterns[a].to_fsm(self._alphabet, self._prefix_postfix)
        except Unsupported as e:
            self._fsms[a] = None
            logger.warning(f"Can't compile Pattern to fsm for {a}\n     {repr(e)}")
        except KeyError:
            self._fsms[a] = None
    return self._fsms[a]