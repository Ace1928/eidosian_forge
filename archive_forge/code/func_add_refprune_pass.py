from ctypes import (c_bool, c_char_p, c_int, c_size_t, c_uint, Structure, byref,
from collections import namedtuple
from enum import IntFlag
from llvmlite.binding import ffi
import os
from tempfile import mkstemp
from llvmlite.binding.common import _encode_string
def add_refprune_pass(self, subpasses_flags=RefPruneSubpasses.ALL, subgraph_limit=1000):
    """Add Numba specific Reference count pruning pass.

        Parameters
        ----------
        subpasses_flags : RefPruneSubpasses
            A bitmask to control the subpasses to be enabled.
        subgraph_limit : int
            Limit the fanout pruners to working on a subgraph no bigger than
            this number of basic-blocks to avoid spending too much time in very
            large graphs. Default is 1000. Subject to change in future
            versions.
        """
    iflags = RefPruneSubpasses(subpasses_flags)
    ffi.lib.LLVMPY_AddRefPrunePass(self, iflags, subgraph_limit)