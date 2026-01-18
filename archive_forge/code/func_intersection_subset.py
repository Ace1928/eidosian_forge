from typing import Any, cast, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Tuple, Union
import itertools
import numpy as np
from cirq import value
from cirq._doc import document
def intersection_subset(self, other: '_QidShapeSet'):
    """Return a subset of the intersection with other qid shape set."""
    explicit_qid_shapes = self.explicit_qid_shapes & other.explicit_qid_shapes
    unfactorized_total_dimension = None
    min_qudit_dimensions = None
    if self.explicit_qid_shapes and other.unfactorized_total_dimension is not None:
        explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(self.explicit_qid_shapes, other.unfactorized_total_dimension)
    if self.explicit_qid_shapes and other.min_qudit_dimensions:
        explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(self.explicit_qid_shapes, other.min_qudit_dimensions)
    if self.unfactorized_total_dimension is not None and other.explicit_qid_shapes:
        explicit_qid_shapes |= _intersection_explicit_with_unfactorized_qid_shapes(other.explicit_qid_shapes, self.unfactorized_total_dimension)
    if self.unfactorized_total_dimension == other.unfactorized_total_dimension:
        unfactorized_total_dimension = self.unfactorized_total_dimension
    if self.min_qudit_dimensions is not None and other.explicit_qid_shapes:
        explicit_qid_shapes |= _intersection_explicit_with_min_qudit_dims_qid_shapes(other.explicit_qid_shapes, self.min_qudit_dimensions)
    if self.min_qudit_dimensions is not None and other.min_qudit_dimensions is not None:
        min_qudit_dimensions = _intersection_min_qudit_dims_qid_shapes(self.min_qudit_dimensions, other.min_qudit_dimensions)
    return _QidShapeSet(explicit_qid_shapes=explicit_qid_shapes, unfactorized_total_dimension=unfactorized_total_dimension, min_qudit_dimensions=min_qudit_dimensions)