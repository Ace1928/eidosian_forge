import sys
from copy import deepcopy
from typing import List, Callable, Iterator, Union, Optional, Generic, TypeVar, TYPE_CHECKING
from collections import OrderedDict
def scan_values(self, pred: 'Callable[[Branch[_Leaf_T]], bool]') -> Iterator[_Leaf_T]:
    """Return all values in the tree that evaluate pred(value) as true.

        This can be used to find all the tokens in the tree.

        Example:
            >>> all_tokens = tree.scan_values(lambda v: isinstance(v, Token))
        """
    for c in self.children:
        if isinstance(c, Tree):
            for t in c.scan_values(pred):
                yield t
        elif pred(c):
            yield c