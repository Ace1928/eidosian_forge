import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np
from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity
def phase_damp(gamma: float) -> PhaseDampingChannel:
    """Creates a PhaseDampingChannel with damping constant gamma.

    This channel evolves a density matrix via:

    $$
    \\rho \\rightarrow M_0 \\rho M_0^\\dagger + M_1 \\rho M_1^\\dagger
    $$

    With:

    $$
    \\begin{aligned}
    M_0 =& \\begin{bmatrix}
            1 & 0  \\\\
            0 & \\sqrt{1 - \\gamma}
          \\end{bmatrix}
    \\\\
    M_1 =& \\begin{bmatrix}
            0 & 0 \\\\
            0 & \\sqrt{\\gamma}
          \\end{bmatrix}
    \\end{aligned}
    $$

    Args:
        gamma: The damping constant.

    Raises:
        ValueError: is gamma is not a valid probability.
    """
    return PhaseDampingChannel(gamma)