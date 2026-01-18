import sys
import io
import copy
import array
import itertools
import struct
import zlib
from collections import namedtuple
from io import BytesIO
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
@classmethod
def calculate_reductions(cls, aveSize):
    bbiMaxZoomLevels = _ZoomLevels.bbiMaxZoomLevels
    reductions = np.zeros(bbiMaxZoomLevels, dtype=[('scale', '=i4'), ('size', '=i4'), ('end', '=i4')])
    minZoom = 10
    res = max(int(aveSize), minZoom)
    maxInt = np.iinfo(reductions.dtype['scale']).max
    for resTry in range(bbiMaxZoomLevels):
        if res > maxInt:
            break
        reductions[resTry]['scale'] = res
        res *= _ZoomLevels.bbiResIncrement
    return reductions[:resTry]