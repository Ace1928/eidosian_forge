import abc
from typing import List, Sequence, Tuple, TYPE_CHECKING
from typing_extensions import Self
import numpy as np
from cirq import value
@property
def can_represent_mixed_states(self) -> bool:
    """Subclasses that can represent mixed states should override this."""
    return False