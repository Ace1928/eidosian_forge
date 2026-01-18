from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def infer_qid_shape(self) -> Optional[Tuple[int, ...]]:
    """Return a qid shape from this set, or None."""
    self._raise_value_error_if_ambiguous()
    if self.unfactorized_total_dimension is not None:
        return _infer_qid_shape_from_dimension(self.unfactorized_total_dimension)
    if len(self.explicit_qid_shapes) == 0:
        return None
    return self.explicit_qid_shapes.pop()