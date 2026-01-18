from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def run_with_remarks(self, function, remarks_format='yaml', remarks_filter=''):
    """
        Run optimization passes on the given function and returns the result
        and the remarks data.

        Parameters
        ----------
        function : llvmlite.binding.FunctionRef
            The function to be optimized inplace
        remarks_format : str; optional
            The format of the remarks file; the default is YAML
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
    remarkdesc, remarkfile = mkstemp()
    try:
        with os.fdopen(remarkdesc, 'r'):
            pass
        r = self.run(function, remarkfile, remarks_format, remarks_filter)
        if r == -1:
            raise IOError('Failed to initialize remarks file.')
        with open(remarkfile) as f:
            return (bool(r), f.read())
    finally:
        os.unlink(remarkfile)