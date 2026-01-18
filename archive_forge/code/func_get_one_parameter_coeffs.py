from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
def get_one_parameter_coeffs(self, interface):
    """Compute the Pauli basis coefficients of the generators of one-parameter groups
        that reproduce the partial derivatives of a special unitary gate.

        Args:
            interface (str): The auto-differentiation framework to be used for the
                computation.

        Returns:
            tensor_like: The Pauli basis coefficients of the effective generators
            that reproduce the partial derivatives of the special unitary gate
            defined by ``theta``. There are :math:`d=4^n-1` generators for
            :math:`n` qubits and :math:`d` Pauli coefficients per generator, so that the
            output shape is ``(4**num_wires-1, 4**num_wires-1)``.

        Given a generator :math:`\\Omega` of a one-parameter group that
        reproduces a partial derivative of a special unitary gate, it can be decomposed in
        the Pauli basis of :math:`\\mathfrak{su}(N)` via

        .. math::

            \\Omega = \\sum_{m=1}^d \\omega_m P_m

        where :math:`d=4^n-1` is the size of the basis for :math:`n` qubits and :math:`P_m` are the
        Pauli words making up the basis. As the Pauli words are orthonormal with respect to the
        `trace or Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`__
        (rescaled by :math:`2^n`), we can compute the coefficients using this
        inner product (:math:`P_m` is Hermitian, so we skip the adjoint :math:`{}^\\dagger`):

        .. math::

            \\omega_m = \\frac{1}{2^n}\\operatorname{tr}\\left[P_m \\Omega \\right]

        The coefficients satisfy :math:`\\omega_m^\\ast=-\\omega_m` because :math:`\\Omega` is
        skew-Hermitian. Therefore they are purely imaginary.

        .. warning::

            An auto-differentiation framework is required by this function.
            The matrix exponential is not differentiable in Autograd. Therefore this function
            only supports JAX, Torch and Tensorflow.

        .. seealso:: :meth:`~.SpecialUnitary.get_one_parameter_generators`

        """
    num_wires = self.hyperparameters['num_wires']
    generators = self.get_one_parameter_generators(interface)
    return _pauli_decompose(generators, num_wires)