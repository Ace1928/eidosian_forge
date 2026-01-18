import itertools
import random
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union
@property
def labelled_edges(self) -> Dict[FrozenSet, Any]:
    return dict(self._labelled_edges)