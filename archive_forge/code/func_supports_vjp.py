from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane.ops.op_math.condition import Conditional
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram
from . import Device
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.simulate import simulate, get_final_state, measure_final_state
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp
def supports_vjp(self, execution_config: Optional[ExecutionConfig]=None, circuit: Optional[QuantumTape]=None) -> bool:
    """Whether or not this device defines a custom vector jacobian product.

        ``DefaultQubit`` supports backpropagation derivatives with analytic results, as well as
        adjoint differentiation.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information
        """
    return self.supports_derivatives(execution_config, circuit)