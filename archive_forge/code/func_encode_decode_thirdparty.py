from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal
import numpy as np
from srsly import msgpack
def encode_decode_thirdparty(self, x, use_bin_type=False, raw=True):
    x_enc = msgpack.packb(x, default=self.encode_thirdparty, use_bin_type=use_bin_type)
    return msgpack.unpackb(x_enc, raw=raw, object_hook=self.decode_thirdparty)