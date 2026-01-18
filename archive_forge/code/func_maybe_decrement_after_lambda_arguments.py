from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_decrement_after_lambda_arguments(self, leaf: Leaf) -> bool:
    """See `maybe_increment_lambda_arguments` above for explanation."""
    if self._lambda_argument_depths and self._lambda_argument_depths[-1] == self.depth and (leaf.type == token.COLON):
        self.depth -= 1
        self._lambda_argument_depths.pop()
        return True
    return False