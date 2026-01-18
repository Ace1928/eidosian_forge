import os
import sys
from mmap import mmap, ACCESS_READ
from mmap import ALLOCATIONGRANULARITY
def align_to_mmap(num, round_up):
    """
    Align the given integer number to the closest page offset, which usually is 4096 bytes.

    :param round_up: if True, the next higher multiple of page size is used, otherwise
        the lower page_size will be used (i.e. if True, 1 becomes 4096, otherwise it becomes 0)
    :return: num rounded to closest page"""
    res = num // ALLOCATIONGRANULARITY * ALLOCATIONGRANULARITY
    if round_up and res != num:
        res += ALLOCATIONGRANULARITY
    return res