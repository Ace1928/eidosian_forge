from functools import lru_cache, reduce
from itertools import product
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.qubit.parametric_ops_multi_qubit import PauliRot
def get_one_parameter_generators(self, interface=None):
    """Compute the generators of one-parameter groups that reproduce
        the partial derivatives of a special unitary gate.

        Args:
            interface (str): The auto-differentiation framework to be used for the
                computation. Has to be one of ``["jax", "tensorflow", "tf", "torch"]``.

        Raises:
            NotImplementedError: If the chosen interface is ``"autograd"``. Autograd
                does not support differentiation of ``linalg.expm``.
            ValueError: If the chosen interface is not supported.

        Returns:
            tensor_like: The generators for one-parameter groups that reproduce the
            partial derivatives of the special unitary gate.
            There are :math:`d=4^n-1` generators for :math:`n` qubits, so that the
            output shape is ``(4**num_wires-1, 2**num_wires, 2**num_wires)``.

        Consider a special unitary gate parametrized in the following way:

        .. math::

            U(\\theta) &= \\exp\\{A(\\theta)\\}\\\\
            A(\\theta) &= \\sum_{m=1}^d i \\theta_m P_m\\\\
            P_m &\\in \\{I, X, Y, Z\\}^{\\otimes n} \\setminus \\{I^{\\otimes n}\\}

        Then the partial derivatives of the gate can be shown to be given by

        .. math::

            \\frac{\\partial}{\\partial \\theta_\\ell} U(\\theta) = U(\\theta)
            \\frac{\\mathrm{d}}{\\mathrm{d}t}\\exp\\left(t\\Omega_\\ell(\\theta)\\right)\\large|_{t=0}

        where :math:`\\Omega_\\ell(\\theta)` is the one-parameter generator belonging to the partial
        derivative :math:`\\partial_\\ell U(\\theta)` at the parameters :math:`\\theta`.
        It can be computed via

        .. math::

            \\Omega_\\ell(\\theta) = U(\\theta)^\\dagger
            \\left(\\frac{\\partial}{\\partial \\theta_\\ell}\\mathfrak{Re}[U(\\theta)]
            +i\\frac{\\partial}{\\partial \\theta_\\ell}\\mathfrak{Im}[U(\\theta)]\\right)

        where we may compute the derivatives :math:`\\frac{\\partial}{\\partial \\theta_\\ell} U(\\theta)` using auto-differentiation.

        .. warning::

            An auto-differentiation framework is required for this function.
            The matrix exponential is not differentiable in Autograd. Therefore this function
            only supports JAX, Torch and TensorFlow.

        """
    theta = self.data[0]
    if len(qml.math.shape(theta)) > 1:
        raise ValueError('Broadcasting is not supported.')
    num_wires = self.hyperparameters['num_wires']

    def split_matrix(theta):
        """Compute the real and imaginary parts of the special unitary matrix."""
        mat = self.compute_matrix(theta, num_wires)
        return (qml.math.real(mat), qml.math.imag(mat))
    if interface == 'jax':
        import jax
        theta = qml.math.cast_like(theta, 1j)
        jac = jax.jacobian(self.compute_matrix, argnums=0, holomorphic=True)(theta, num_wires)
    elif interface == 'torch':
        import torch
        rjac, ijac = torch.autograd.functional.jacobian(split_matrix, theta)
        jac = rjac + 1j * ijac
    elif interface in ('tensorflow', 'tf'):
        import tensorflow as tf
        with tf.GradientTape(persistent=True) as tape:
            mats = qml.math.stack(split_matrix(theta))
        rjac, ijac = tape.jacobian(mats, theta)
        jac = qml.math.cast_like(rjac, 1j) + 1j * qml.math.cast_like(ijac, 1j)
    elif interface == 'autograd':
        raise NotImplementedError('The matrix exponential expm is not differentiable in Autograd.')
    else:
        raise ValueError(f'The interface {interface} is not supported.')
    U_dagger = self.compute_matrix(-qml.math.detach(theta), num_wires)
    return qml.math.transpose(qml.math.tensordot(U_dagger, jac, axes=[[1], [0]]), [2, 0, 1])