import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
@property
def projector_dict(self) -> Dict[raw_types.Qid, int]:
    return self._projector_dict