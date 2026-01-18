from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import operator
import warnings
from functools import reduce
import numpy as np
from numba.np.ufunc.ufuncbuilder import _BaseUFuncBuilder, parse_identity
from numba.core import types, sigutils
from numba.core.typing import signature
from numba.np.ufunc.sigparse import parse_signature
def prepare_outputs(self, schedule, outdtypes):
    """
        Returns a list of output parameters that all reside on the target device.

        Outputs that were passed-in to the GUFunc are used if they reside on the
        device; other outputs are allocated as necessary.
        """
    outputs = []
    for shape, dtype, output in zip(schedule.output_shapes, outdtypes, self.outputs):
        if output is None or self._copy_result_to_host:
            output = self.allocate_device_array(shape, dtype)
        outputs.append(output)
    return outputs