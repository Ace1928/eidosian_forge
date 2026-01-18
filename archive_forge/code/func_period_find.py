import math
import random
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.numbers import igcd
from sympy.ntheory import continued_fraction_periodic as continued_fraction
from sympy.utilities.iterables import variations
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import Qubit, measure_partial_oneshot
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qft import QFT
from sympy.physics.quantum.qexpr import QuantumError
def period_find(a, N):
    """Finds the period of a in modulo N arithmetic

    This is quantum part of Shor's algorithm. It takes two registers,
    puts first in superposition of states with Hadamards so: ``|k>|0>``
    with k being all possible choices. It then does a controlled mod and
    a QFT to determine the order of a.
    """
    epsilon = 0.5
    t = int(2 * math.ceil(log(N, 2)))
    start = [0 for x in range(t)]
    factor = 1 / sqrt(2 ** t)
    qubits = 0
    for arr in variations(range(2), t, repetition=True):
        qbitArray = list(arr) + start
        qubits = qubits + Qubit(*qbitArray)
    circuit = (factor * qubits).expand()
    circuit = CMod(t, a, N) * circuit
    circuit = qapply(circuit)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i)
    circuit = qapply(QFT(t, t * 2).decompose() * circuit, floatingPoint=True)
    for i in range(t):
        circuit = measure_partial_oneshot(circuit, i + t)
    if isinstance(circuit, Qubit):
        register = circuit
    elif isinstance(circuit, Mul):
        register = circuit.args[-1]
    else:
        register = circuit.args[-1].args[-1]
    n = 1
    answer = 0
    for i in range(len(register) / 2):
        answer += n * register[i + t]
        n = n << 1
    if answer == 0:
        raise OrderFindingException('Order finder returned 0. Happens with chance %f' % epsilon)
    g = getr(answer, 2 ** t, N)
    return g