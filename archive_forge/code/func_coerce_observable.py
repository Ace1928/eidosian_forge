from __future__ import annotations
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping as _Mapping
from functools import lru_cache
from typing import Union, Mapping, overload
from numbers import Complex
import numpy as np
from numpy.typing import ArrayLike
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp
from .object_array import object_array
from .shape import ShapedMixin, shape_tuple
@classmethod
def coerce_observable(cls, observable: ObservableLike) -> Mapping[str, float]:
    """Format an observable-like object into the internal format.

        Args:
            observable: The observable-like to format.

        Returns:
            The coerced observable.

        Raises:
            TypeError: If the input cannot be formatted because its type is not valid.
            ValueError: If the input observable is invalid.
        """
    if isinstance(observable, SparsePauliOp):
        observable = observable.simplify(atol=0)
        coeffs = np.real_if_close(observable.coeffs)
        if np.iscomplexobj(coeffs):
            raise ValueError('Non-Hermitian input observable: the input SparsePauliOp has non-zero imaginary part in its coefficients.')
        paulis = observable.paulis.to_labels()
        return dict(zip(paulis, coeffs))
    if isinstance(observable, Pauli):
        label, phase = (observable[:].to_label(), observable.phase)
        if phase % 2:
            raise ValueError('Non-Hermitian input observable: the input Pauli has an imaginary phase.')
        return {label: 1} if phase == 0 else {label: -1}
    if isinstance(observable, str):
        cls._validate_basis(observable)
        return {observable: 1}
    if isinstance(observable, _Mapping):
        num_qubits = len(next(iter(observable)))
        unique = defaultdict(float)
        for basis, coeff in observable.items():
            if isinstance(basis, Pauli):
                basis, phase = (basis[:].to_label(), basis.phase)
                if phase % 2:
                    raise ValueError('Non-Hermitian input observable: the input Pauli has an imaginary phase.')
                if phase == 2:
                    coeff = -coeff
            if isinstance(coeff, Complex):
                if abs(coeff.imag) > 1e-07:
                    raise TypeError(f'Non-Hermitian input observable: {basis} term has a complex value coefficient.')
                coeff = coeff.real
            cls._validate_basis(basis)
            if len(basis) != num_qubits:
                raise ValueError('Number of qubits must be the same for all observable basis elements.')
            unique[basis] += coeff
        return dict(unique)
    raise TypeError(f'Invalid observable type: {type(observable)}')