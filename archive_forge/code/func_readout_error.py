import copy
import datetime
from typing import Any, Iterable, Tuple, Union, Dict
import dateutil.parser
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.utils.units import apply_prefix
def readout_error(self, qubit: int) -> float:
    """
        Return the readout error of the given qubit.

        Args:
            qubit: Qubit for which to return the readout error of.

        Return:
            Readout error of the given qubit.
        """
    return self.qubit_property(qubit, 'readout_error')[0]