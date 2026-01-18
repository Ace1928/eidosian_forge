import pennylane as qml
from pennylane.operation import Tensor
def mixer_layer(alpha, hamiltonian):
    """Applies the QAOA mixer layer corresponding to a mixer Hamiltonian.

    For a mixer Hamiltonian :math:`H_M`, this is defined as the following unitary:

    .. math:: U_M \\ = \\ e^{-i \\alpha H_M}

    where :math:`\\alpha` is a variational parameter.

    Args:
        alpha (int or float): The variational parameter passed into the mixer layer
        hamiltonian (.Hamiltonian): The mixer Hamiltonian

    .. details::
        :title: Usage Details

        We first define a mixer Hamiltonian:

        .. code-block:: python3

            from pennylane import qaoa
            import pennylane as qml

            mixer_h = qml.Hamiltonian([1, 1], [qml.X(0), qml.X(0) @ qml.X(1)])

        We can then pass it into ``qaoa.mixer_layer``, within a quantum circuit:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=2)

            @qml.qnode(dev)
            def circuit(alpha):

                for i in range(2):
                    qml.Hadamard(wires=i)

                qaoa.mixer_layer(alpha, mixer_h)

                return [qml.expval(qml.Z(i)) for i in range(2)]

        which gives us a circuit of the form:

        >>> print(qml.draw(circuit)(0.5))
        0: ──H─╭ApproxTimeEvolution(1.00,1.00,0.50)─┤  <Z>
        1: ──H─╰ApproxTimeEvolution(1.00,1.00,0.50)─┤  <Z>
        >>> print(qml.draw(circuit, expansion_strategy="device")(0.5))
        0: ──H──RX(1.00)─╭RXX(1.00)─┤  <Z>
        1: ──H───────────╰RXX(1.00)─┤  <Z>

    """
    if not isinstance(hamiltonian, qml.Hamiltonian):
        raise ValueError(f'hamiltonian must be of type pennylane.Hamiltonian, got {type(hamiltonian).__name__}')
    qml.templates.ApproxTimeEvolution(hamiltonian, alpha, 1)