import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core import internals
from pandas.core.internals import api
def test_create_block_manager_from_blocks_deprecated():
    msg = 'create_block_manager_from_blocks is deprecated and will be removed in a future version. Use public APIs instead'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        internals.create_block_manager_from_blocks