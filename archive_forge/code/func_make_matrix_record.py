import os
import zlib
import time  # noqa
import logging
import numpy as np
def make_matrix_record(self, scale_xy=None, rot_xy=None, trans_xy=None):
    if scale_xy is None and rot_xy is None and (trans_xy is None):
        return '0' * 8
    bits = BitArray()
    if scale_xy:
        bits += '1'
        bits += floats2bits([scale_xy[0], scale_xy[1]])
    else:
        bits += '0'
    if rot_xy:
        bits += '1'
        bits += floats2bits([rot_xy[0], rot_xy[1]])
    else:
        bits += '0'
    if trans_xy:
        bits += twits2bits([trans_xy[0], trans_xy[1]])
    else:
        bits += twits2bits([0, 0])
    return bits