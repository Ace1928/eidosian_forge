import itertools
import math
from typing import (
import numpy
from ..types import (
from ..util import get_array_module, is_xp_array, to_numpy
from .cblas import CBlas
def backprop_lstm_gates(dYt3: Floats2d, dCt3: Floats2d, Gt3: Floats2d, Ct3: Floats2d, Ct2: Floats2d) -> Tuple[Floats2d, Floats2d]:
    xp = get_array_module(dYt3)
    hf, hi, ho, hc = xp.split(Gt3, 4, axis=-1)
    assert hf.shape[0] == hi.shape[0] == ho.shape[0] == hc.shape[0]
    assert hf.shape[0] == dYt3.shape[0] == dCt3.shape[0] == Ct3.shape[0] == Ct2.shape[0]
    tanhCt3 = xp.tanh(Ct3)
    d_ho = dYt3 * tanhCt3
    d_tanhCt3 = dYt3 * ho
    dCt3 += d_tanhCt3 * dtanh(tanhCt3)
    d_hi = dCt3 * hc
    d_hc = dCt3 * hi
    d_hf = dCt3 * Ct2
    dCt2 = dCt3 * hf
    d_At3_hc = d_hc * dtanh(hc)
    d_At3_ho = d_ho * dsigmoid(ho)
    d_At3_hi = d_hi * dsigmoid(hi)
    d_At3_hf = d_hf * dsigmoid(hf)
    dAt3 = xp.concatenate((d_At3_hf, d_At3_hi, d_At3_ho, d_At3_hc), axis=-1)
    return (dAt3, dCt2)