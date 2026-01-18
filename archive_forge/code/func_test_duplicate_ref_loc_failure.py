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
def test_duplicate_ref_loc_failure(self):
    tmp_mgr = create_mgr('a:bool; a: f8')
    axes, blocks = (tmp_mgr.axes, tmp_mgr.blocks)
    blocks[0].mgr_locs = BlockPlacement(np.array([0]))
    blocks[1].mgr_locs = BlockPlacement(np.array([0]))
    msg = 'Gaps in blk ref_locs'
    with pytest.raises(AssertionError, match=msg):
        mgr = BlockManager(blocks, axes)
        mgr._rebuild_blknos_and_blklocs()
    blocks[0].mgr_locs = BlockPlacement(np.array([0]))
    blocks[1].mgr_locs = BlockPlacement(np.array([1]))
    mgr = BlockManager(blocks, axes)
    mgr.iget(1)