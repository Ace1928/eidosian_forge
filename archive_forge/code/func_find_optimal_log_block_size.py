from typing import List, Optional, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import qrom, swap_network
def find_optimal_log_block_size(iteration_length: int, target_bitsize: int) -> int:
    """Find optimal block size, which is a power of 2, for SelectSwapQROM.

    This functions returns the optimal `k` s.t.
        * k is in an integer and k >= 0.
        * iteration_length/2^k + target_bitsize*(2^k - 1) is minimized.
    The corresponding block size for SelectSwapQROM would be 2^k.
    """
    k = 0.5 * np.log2(iteration_length / target_bitsize)
    if k < 0:
        return 1

    def value(kk: List[int]):
        return iteration_length / np.power(2, kk) + target_bitsize * (np.power(2, kk) - 1)
    k_int = [np.floor(k), np.ceil(k)]
    return int(k_int[np.argmin(value(k_int))])