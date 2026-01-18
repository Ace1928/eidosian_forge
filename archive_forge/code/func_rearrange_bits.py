import os
import json
import struct
import logging
import numpy as np
from ..core import Format
from ..v2 import imread
@staticmethod
def rearrange_bits(array):
    t0 = array[0::3]
    t1 = array[1::3]
    t2 = array[2::3]
    a0 = np.left_shift(t0, 4) + np.right_shift(np.bitwise_and(t1, 240), 4)
    a1 = np.left_shift(np.bitwise_and(t1, 15), 8) + t2
    image = np.zeros(LYTRO_F01_IMAGE_SIZE, dtype=np.uint16)
    image[:, 0::2] = a0.reshape((LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2))
    image[:, 1::2] = a1.reshape((LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2))
    return np.divide(image, 4095.0).astype(np.float64)