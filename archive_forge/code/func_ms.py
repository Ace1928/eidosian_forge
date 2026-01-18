from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Sequence
from typing_extensions import Self
import numpy as np
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq._doc import document
from cirq.ops import (
def ms(rads: float) -> MSGate:
    """A helper to construct the `cirq.MSGate` for the given angle specified in radians.

    Args:
        rads: The rotation angle in radians.

    Returns:
        Mølmer–Sørensen gate rotating by the desired amount.
    """
    return MSGate(rads=rads)