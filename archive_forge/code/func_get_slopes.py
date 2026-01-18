import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def get_slopes(n: int):

    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-2 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]