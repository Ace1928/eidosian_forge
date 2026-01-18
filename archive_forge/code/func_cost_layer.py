import pennylane as qml
from pennylane.operation import Tensor
def cost_layer(gamma, hamiltonian):
    """Applies the QAOA cost layer corresponding to a cost Hamiltonian.

    For the cost Hamiltonian :math:`H_C`, this is defined as the following unitary:

    .. math:: U_C \\ = \\ e^{-i \\gamma H_C}

    where :math:`\\gamma` is a variational parameter.

    Args:
        gamma (int or float): The variational parameter passed into the cost layer
        hamiltonian (.Hamiltonian): The cost Hamiltonian

    Raises:
        ValueError: if the terms of the supplied cost Hamiltonian are not exclusively products of diagonal Pauli gates

    .. details::
        :title: Usage Details

        We first define a cost Hamiltonian:

        .. code-block:: python3

            from pennylane import qaoa
            import pennylane as qml

            cost_h = qml.Hamiltonian([1, 1], [qml.Z(0), qml.Z(0) @ qml.Z(1)])

        We can then pass it into ``qaoa.cost_layer``, within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(gamma):

                for i in range(2):
                    qml.Hadamard(wires=i)

                qaoa.cost_layer(gamma, cost_h)

                return [qml.expval(qml.Z(i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> print(qml.draw(circuit)(0.5))
        0: ──H─╭ApproxTimeEvolution(1.00,1.00,0.50)─┤  <Z>
        1: ──H─╰ApproxTimeEvolution(1.00,1.00,0.50)─┤  <Z>
        >>> print(qml.draw(circuit, expansion_strategy="device")(0.5))
        0: ──H──RZ(1.00)─╭RZZ(1.00)─┤  <Z>
        1: ──H───────────╰RZZ(1.00)─┤  <Z>

    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(f'hamiltonian must be of type pennylane.Hamiltonian, got {type(hamiltonian).__name__}')
    if not _diagonal_terms(hamiltonian):
        raise ValueError('hamiltonian must be written only in terms of PauliZ and Identity gates')
    qml.templates.ApproxTimeEvolution(hamiltonian, gamma, 1)