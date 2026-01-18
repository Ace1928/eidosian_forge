from collections import defaultdict
import copy
import itertools
import os
import linecache
import pprint
import re
import sys
import operator
from types import FunctionType, BuiltinFunctionType
from functools import total_ordering
from io import StringIO
from numba.core import errors, config
from numba.core.utils import (BINOPS_TO_OPERATORS, INPLACE_BINOPS_TO_OPERATORS,
from numba.core.errors import (NotDefinedError, RedefinedError,
from numba.core import consts
def render_dot(self, filename_prefix='numba_ir', include_ir=True):
    """Render the CFG of the IR with GraphViz DOT via the
        ``graphviz`` python binding.

        Returns
        -------
        g : graphviz.Digraph
            Use `g.view()` to open the graph in the default PDF application.
        """
    try:
        import graphviz as gv
    except ImportError:
        raise ImportError('The feature requires `graphviz` but it is not available. Please install with `pip install graphviz`')
    g = gv.Digraph(filename='{}{}.dot'.format(filename_prefix, self.func_id.unique_name))
    for k, blk in self.blocks.items():
        with StringIO() as sb:
            blk.dump(sb)
            label = sb.getvalue()
        if include_ir:
            label = ''.join(['  {}\\l'.format(x) for x in label.splitlines()])
            label = 'block {}\\l'.format(k) + label
            g.node(str(k), label=label, shape='rect')
        else:
            label = '{}\\l'.format(k)
            g.node(str(k), label=label, shape='circle')
    for src, blk in self.blocks.items():
        for dst in blk.terminator.get_targets():
            g.edge(str(src), str(dst))
    return g