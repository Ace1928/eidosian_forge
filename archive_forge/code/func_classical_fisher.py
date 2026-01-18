from functools import partial
from typing import Callable, Sequence
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.devices import DefaultQubit, DefaultQubitLegacy, DefaultMixed
from pennylane.measurements import StateMP, DensityMatrixMP
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane import transform
def classical_fisher(qnode, argnums=0):
    """Returns a function that computes the classical fisher information matrix (CFIM) of a given :class:`.QNode` or
    quantum tape.

    Given a parametrized (classical) probability distribution :math:`p(\\bm{\\theta})`, the classical fisher information
    matrix quantifies how changes to the parameters :math:`\\bm{\\theta}` are reflected in the probability distribution.
    For a parametrized quantum state, we apply the concept of classical fisher information to the computational
    basis measurement.
    More explicitly, this function implements eq. (15) in `arxiv:2103.15191 <https://arxiv.org/abs/2103.15191>`_:

    .. math::

        \\text{CFIM}_{i, j} = \\sum_{\\ell=0}^{2^N-1} \\frac{1}{p_\\ell(\\bm{\\theta})} \\frac{\\partial p_\\ell(\\bm{\\theta})}{
        \\partial \\theta_i} \\frac{\\partial p_\\ell(\\bm{\\theta})}{\\partial \\theta_j}

    for :math:`N` qubits.

    Args:
        tape (:class:`.QNode` or qml.QuantumTape): A :class:`.QNode` or quantum tape that may have arbitrary return types.
        argnums (Optional[int or List[int]]): Arguments to be differentiated in case interface ``jax`` is used.

    Returns:
        func: The function that computes the classical fisher information matrix. This function accepts the same
        signature as the :class:`.QNode`. If the signature contains one differentiable variable ``params``, the function
        returns a matrix of size ``(len(params), len(params))``. For multiple differentiable arguments ``x, y, z``,
        it returns a list of sizes ``[(len(x), len(x)), (len(y), len(y)), (len(z), len(z))]``.


    .. seealso:: :func:`~.pennylane.metric_tensor`, :func:`~.pennylane.qinfo.transforms.quantum_fisher`

    **Example**

    First, let us define a parametrized quantum state and return its (classical) probability distribution for all
    computational basis elements:

    .. code-block:: python

        import pennylane.numpy as pnp
        n_wires = 2

        dev = qml.device("default.qubit", wires=n_wires)

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RX(params[1], wires=0)
            qml.CNOT(wires=(0,1))
            return qml.probs(wires=range(n_wires))

    Executing this circuit yields the ``2**n_wires`` elements of :math:`p_\\ell(\\bm{\\theta})`

    >>> params = pnp.random.random(2)
    >>> circ(params)
    [0.61281668 0.         0.         0.38718332]

    We can obtain its ``(2, 2)`` classical fisher information matrix (CFIM) by simply calling the function returned
    by ``classical_fisher()``:

    >>> cfim_func = qml.qinfo.classical_fisher(circ)
    >>> cfim_func(params)
    [[1. 1.]
     [1. 1.]]

    This function has the same signature as the :class:`.QNode`. Here is a small example with multiple arguments:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.probs(wires=range(n_wires))

    >>> x, y = pnp.array([0.5, 0.6], requires_grad=True)
    >>> circ(x, y)
    [0.86215007 0.         0.13784993 0.        ]
    >>> qml.qinfo.classical_fisher(circ)(x, y)
    [array([[0.32934729]]), array([[0.51650396]])]

    Note how in the case of multiple variables we get a list of matrices with sizes
    ``[(n_params0, n_params0), (n_params1, n_params1)]``, which in this case is simply two ``(1, 1)`` matrices.


    A typical setting where the classical fisher information matrix is used is in variational quantum algorithms.
    Closely related to the `quantum natural gradient <https://arxiv.org/abs/1909.02108>`_, which employs the
    `quantum` fisher information matrix, we can compute a rescaled gradient using the CFIM. In this scenario,
    typically a Hamiltonian objective function :math:`\\langle H \\rangle` is minimized:

    .. code-block:: python

        H = qml.Hamiltonian(coeffs=[0.5, 0.5], observables=[qml.Z(0), qml.Z(1)])

        @qml.qnode(dev)
        def circ(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RX(params[2], wires=1)
            qml.RY(params[3], wires=1)
            qml.CNOT(wires=(0,1))
            return qml.expval(H)

        params = pnp.random.random(4)

    We can compute both the gradient of :math:`\\langle H \\rangle` and the CFIM with the same :class:`.QNode` ``circ``
    in this example since ``classical_fisher()`` ignores the return types and assumes ``qml.probs()`` for all wires.

    >>> grad = qml.grad(circ)(params)
    >>> cfim = qml.qinfo.classical_fisher(circ)(params)
    >>> print(grad.shape, cfim.shape)
    (4,) (4, 4)

    Combined together, we can get a rescaled gradient to be employed for optimization schemes like natural gradient
    descent.

    >>> rescaled_grad = cfim @ grad
    >>> print(rescaled_grad)
    [-0.66772533 -0.16618756 -0.05865127 -0.06696078]

    The ``classical_fisher`` matrix itself is again differentiable:

    .. code-block:: python

        @qml.qnode(dev)
        def circ(params):
            qml.RX(qml.math.cos(params[0]), wires=0)
            qml.RX(qml.math.cos(params[0]), wires=1)
            qml.RX(qml.math.cos(params[1]), wires=0)
            qml.RX(qml.math.cos(params[1]), wires=1)
            return qml.probs(wires=range(2))

        params = pnp.random.random(2)

    >>> qml.qinfo.classical_fisher(circ)(params)
    [[4.18575068e-06 2.34443943e-03]
     [2.34443943e-03 1.31312079e+00]]
    >>> qml.jacobian(qml.qinfo.classical_fisher(circ))(params)
    array([[[9.98030491e-01, 3.46944695e-18],
            [1.36541817e-01, 5.15248592e-01]],
           [[1.36541817e-01, 5.15248592e-01],
            [2.16840434e-18, 2.81967252e-01]]]))

    """
    new_qnode = _make_probs(qnode)

    def wrapper(*args, **kwargs):
        old_interface = qnode.interface
        if old_interface == 'auto':
            qnode.interface = qml.math.get_interface(*args, *list(kwargs.values()))
        interface = qnode.interface
        if interface in ('jax', 'jax-jit'):
            import jax
            jac = jax.jacobian(new_qnode, argnums=argnums)
        if interface == 'torch':
            jac = _torch_jac(new_qnode)
        if interface == 'autograd':
            jac = qml.jacobian(new_qnode)
        if interface == 'tf':
            jac = _tf_jac(new_qnode)
        j = jac(*args, **kwargs)
        p = new_qnode(*args, **kwargs)
        if old_interface == 'auto':
            qnode.interface = 'auto'
        if isinstance(j, tuple):
            res = []
            for j_i in j:
                res.append(_compute_cfim(p, j_i))
            if len(j) == 1:
                return res[0]
            return res
        return _compute_cfim(p, j)
    return wrapper