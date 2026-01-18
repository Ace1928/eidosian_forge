from typing import Any, Dict, List, Sequence, Union
import numpy as np
import cirq
from cirq import protocols, qis, value
from cirq.value import big_endian_int_to_digits, random_state
def inner_product_of_state_and_x(self, x: int) -> Union[float, complex]:
    """Returns the amplitude of x'th element of
        the state vector, i.e. <x|psi>"""
    if type(x) == int:
        y = cirq.big_endian_int_to_bits(x, bit_count=self.n)
    mu = sum(y * self.gamma)
    u = np.zeros(self.n, dtype=bool)
    for p in range(self.n):
        if y[p]:
            u ^= self.F[p, :]
            mu += 2 * (sum(self.M[p, :] & u) % 2)
    return self.omega * 2 ** (-sum(self.v) / 2) * 1j ** mu * (-1) ** sum(self.v & u & self.s) * bool(np.all(self.v | (u == self.s)))