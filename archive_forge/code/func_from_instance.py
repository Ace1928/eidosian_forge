import hashlib
import re
import warnings
from enum import Enum
from functools import singledispatchmethod
from typing import Union, List, Iterator, Optional
import numpy as np
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse import channels, instructions, library
from qiskit.pulse.configuration import Kernel, Discriminator
from qiskit.pulse.exceptions import QiskitError
from qiskit.pulse.parser import parse_string_expr
from qiskit.pulse.schedule import Schedule
from qiskit.qobj import QobjMeasurementOption, PulseLibraryItem, PulseQobjInstruction
from qiskit.qobj.utils import MeasLevel
@classmethod
def from_instance(cls, instance: library.SymbolicPulse) -> 'ParametricPulseShapes':
    """Get Qobj name from the pulse class instance.

        Args:
            instance: SymbolicPulse class.

        Returns:
            Qobj name.

        Raises:
            QiskitError: When pulse instance is not recognizable type.
        """
    if isinstance(instance, library.SymbolicPulse):
        return cls(instance.pulse_type)
    raise QiskitError(f"'{instance}' is not valid pulse type.")