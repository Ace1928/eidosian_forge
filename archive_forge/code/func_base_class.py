from __future__ import annotations
import copy
from itertools import zip_longest
import math
from typing import List, Type
import numpy
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
@property
def base_class(self) -> Type[Instruction]:
    """Get the base class of this instruction.  This is guaranteed to be in the inheritance tree
        of ``self``.

        The "base class" of an instruction is the lowest class in its inheritance tree that the
        object should be considered entirely compatible with for _all_ circuit applications.  This
        typically means that the subclass is defined purely to offer some sort of programmer
        convenience over the base class, and the base class is the "true" class for a behavioural
        perspective.  In particular, you should *not* override :attr:`base_class` if you are
        defining a custom version of an instruction that will be implemented differently by
        hardware, such as an alternative measurement strategy, or a version of a parametrised gate
        with a particular set of parameters for the purposes of distinguishing it in a
        :class:`.Target` from the full parametrised gate.

        This is often exactly equivalent to ``type(obj)``, except in the case of singleton instances
        of standard-library instructions.  These singleton instances are special subclasses of their
        base class, and this property will return that base.  For example::

            >>> isinstance(XGate(), XGate)
            True
            >>> type(XGate()) is XGate
            False
            >>> XGate().base_class is XGate
            True

        In general, you should not rely on the precise class of an instruction; within a given
        circuit, it is expected that :attr:`Instruction.name` should be a more suitable
        discriminator in most situations.
        """
    return type(self)