import copy
from collections.abc import Iterable
from typing import Optional, Union, Sequence
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from .measurements import MeasurementShapeError, MeasurementTransform, Shadow, ShadowExpval
def process_state_with_shots(self, state: Sequence[complex], wire_order: Wires, shots: int, rng=None):
    """Process the given quantum state with the given number of shots

        Args:
            state (Sequence[complex]): quantum state
            wire_order (Wires): wires determining the subspace that ``state`` acts on; a matrix of
                dimension :math:`2^n` acts on a subspace of :math:`n` wires
            shots (int): The number of shots
            rng (Union[None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
                seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``.
                If no value is provided, a default RNG will be used.

        Returns:
            float: The estimate of the expectation value.
        """
    bits, recipes = qml.classical_shadow(wires=self.wires, seed=self.seed).process_state_with_shots(state, wire_order, shots, rng=rng)
    shadow = qml.shadows.ClassicalShadow(bits, recipes, wire_map=self.wires.tolist())
    return shadow.expval(self.H, self.k)