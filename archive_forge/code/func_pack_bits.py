from typing import cast, Dict, Hashable, Iterable, List, Optional, Sequence
from collections import OrderedDict
import dataclasses
import numpy as np
import cirq
from cirq_google.api import v2
from cirq_google.api.v2 import result_pb2
def pack_bits(bits: np.ndarray) -> bytes:
    """Pack bits given as a numpy array of bools into bytes."""
    pad = -len(bits) % 8
    if pad:
        bits = np.pad(bits, (0, pad), 'constant')
    bits = bits.reshape((-1, 8))[:, ::-1]
    byte_arr = np.packbits(bits, axis=1).reshape(-1)
    return byte_arr.tobytes()