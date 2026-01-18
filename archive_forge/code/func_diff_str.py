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
def diff_str(self, other):
    """
        Compute a human readable difference in the IR, returns a formatted
        string ready for printing.
        """
    msg = []
    for label, block in self.blocks.items():
        other_blk = other.blocks.get(label, None)
        if other_blk is not None:
            if block != other_blk:
                msg.append(('Block %s differs' % label).center(80, '-'))
                block_del = [x for x in block.body if isinstance(x, Del)]
                oth_del = [x for x in other_blk.body if isinstance(x, Del)]
                if block_del != oth_del:
                    if sorted(block_del) == sorted(oth_del):
                        msg.append('Block %s contains the same dels but their order is different' % label)
                if len(block.body) > len(other_blk.body):
                    msg.append('This block contains more statements')
                elif len(block.body) < len(other_blk.body):
                    msg.append('Other block contains more statements')
                tmp = []
                for idx, stmts in enumerate(zip(block.body, other_blk.body)):
                    b_s, o_s = stmts
                    if b_s != o_s:
                        tmp.append(idx)

                def get_pad(ablock, l):
                    pointer = '-> '
                    sp = len(pointer) * ' '
                    pad = []
                    nstmt = len(ablock)
                    for i in range(nstmt):
                        if i in tmp:
                            item = pointer
                        elif i >= l:
                            item = pointer
                        else:
                            item = sp
                        pad.append(item)
                    return pad
                min_stmt_len = min(len(block.body), len(other_blk.body))
                with StringIO() as buf:
                    it = [('self', block), ('other', other_blk)]
                    for name, _block in it:
                        buf.truncate(0)
                        _block.dump(file=buf)
                        stmts = buf.getvalue().splitlines()
                        pad = get_pad(_block.body, min_stmt_len)
                        title = '%s: block %s' % (name, label)
                        msg.append(title.center(80, '-'))
                        msg.extend(['{0}{1}'.format(a, b) for a, b in zip(pad, stmts)])
    if msg == []:
        msg.append('IR is considered equivalent.')
    return '\n'.join(msg)