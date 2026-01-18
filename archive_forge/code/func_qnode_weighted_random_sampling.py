from copy import copy
import numpy as np
from scipy.stats import multinomial
import pennylane as qml
from .gradient_descent import GradientDescentOptimizer
@staticmethod
def qnode_weighted_random_sampling(qnode, coeffs, observables, shots, argnums, *args, **kwargs):
    """Returns an array of length ``shots`` containing single-shot estimates
        of the Hamiltonian gradient. The shots are distributed randomly over
        the terms in the Hamiltonian, as per a multinomial distribution.

        Args:
            qnode (.QNode): A QNode that returns the expectation value of a Hamiltonian.
            coeffs (List[float]): The coefficients of the Hamiltonian being measured
            observables (List[Observable]): The terms of the Hamiltonian being measured
            shots (int): The number of shots used to estimate the Hamiltonian expectation
                value. These shots are distributed over the terms in the Hamiltonian,
                as per a Multinomial distribution.
            argnums (Sequence[int]): the QNode argument indices which are trainable
            *args: Arguments to the QNode
            **kwargs: Keyword arguments to the QNode

        Returns:
            array[float]: the single-shot gradients of the Hamiltonian expectation value
        """
    qnode = copy(qnode)
    base_func = qnode.func
    prob_shots = np.abs(coeffs) / np.sum(np.abs(coeffs))
    si = multinomial(n=shots, p=prob_shots)
    shots_per_term = si.rvs()[0]
    grads = []
    for o, c, p, s in zip(observables, coeffs, prob_shots, shots_per_term):
        if s == 0:
            continue

        def func(*qnode_args, **qnode_kwargs):
            qs = qml.tape.make_qscript(base_func)(*qnode_args, **qnode_kwargs)
            for op in qs.operations:
                qml.apply(op)
            return qml.expval(o)
        qnode.func = func
        new_shots = 1 if s == 1 else [(1, int(s))]
        if s > 1:

            def cost(*args, **kwargs):
                return qml.math.stack(qnode(*args, **kwargs))
        else:
            cost = qnode
        jacs = qml.jacobian(cost, argnum=argnums)(*args, **kwargs, shots=new_shots)
        if s == 1:
            jacs = [np.expand_dims(j, 0) for j in jacs]
        grads.append([c * j / p for j in jacs])
    return [np.concatenate(i) for i in zip(*grads)]