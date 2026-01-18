from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
def create_mgr(descr, item_shape=None):
    """
    Construct BlockManager from string description.

    String description syntax looks similar to np.matrix initializer.  It looks
    like this::

        a,b,c: f8; d,e,f: i8

    Rules are rather simple:

    * see list of supported datatypes in `create_block` method
    * components are semicolon-separated
    * each component is `NAME,NAME,NAME: DTYPE_ID`
    * whitespace around colons & semicolons are removed
    * components with same DTYPE_ID are combined into single block
    * to force multiple blocks with same dtype, use '-SUFFIX'::

        'a:f8-1; b:f8-2; c:f8-foobar'

    """
    if item_shape is None:
        item_shape = (N,)
    offset = 0
    mgr_items = []
    block_placements = {}
    for d in descr.split(';'):
        d = d.strip()
        if not len(d):
            continue
        names, blockstr = d.partition(':')[::2]
        blockstr = blockstr.strip()
        names = names.strip().split(',')
        mgr_items.extend(names)
        placement = list(np.arange(len(names)) + offset)
        try:
            block_placements[blockstr].extend(placement)
        except KeyError:
            block_placements[blockstr] = placement
        offset += len(names)
    mgr_items = Index(mgr_items)
    blocks = []
    num_offset = 0
    for blockstr, placement in block_placements.items():
        typestr = blockstr.split('-')[0]
        blocks.append(create_block(typestr, placement, item_shape=item_shape, num_offset=num_offset))
        num_offset += len(placement)
    sblocks = sorted(blocks, key=lambda b: b.mgr_locs[0])
    return BlockManager(tuple(sblocks), [mgr_items] + [Index(np.arange(n)) for n in item_shape])