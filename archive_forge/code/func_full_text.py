from collections import namedtuple
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Any, Dict, Iterable, Set, FrozenSet, Optional
from interegular import InvalidSyntax, REFlags
from interegular.fsm import FSM, Alphabet, anything_else
from interegular.patterns import Pattern, Unsupported, parse_pattern
from interegular.utils import logger, soft_repr
@property
def full_text(self):
    return self.prefix + self.main_text + self.postfix