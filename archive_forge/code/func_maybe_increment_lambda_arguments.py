from dataclasses import dataclass, field
from typing import Dict, Final, Iterable, List, Optional, Sequence, Set, Tuple, Union
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def maybe_increment_lambda_arguments(self, leaf: Leaf) -> bool:
    """In a lambda expression, there might be more than one argument.

        To avoid splitting on the comma in this situation, increase the depth of
        tokens between `lambda` and `:`.
        """
    if leaf.type == token.NAME and leaf.value == 'lambda':
        self.depth += 1
        self._lambda_argument_depths.append(self.depth)
        return True
    return False