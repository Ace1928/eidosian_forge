import enum
from typing import Any, List, Optional, TYPE_CHECKING, Union
import pandas as pd
import sympy
from matplotlib import pyplot as plt
from cirq import circuits, ops, study, value
from cirq._compat import proper_repr
@property
def expectation_pauli_x(self) -> pd.DataFrame:
    """A data frame with delay_ns, value columns.

        This value contains the expectation of the Pauli X operator as
        estimated by measurement outcomes.
        """
    return self._expectation_pauli_x