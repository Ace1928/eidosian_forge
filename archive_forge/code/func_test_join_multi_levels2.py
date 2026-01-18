import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
def test_join_multi_levels2(self):
    household = DataFrame({'household_id': [1, 2, 2, 3, 3, 3, 4], 'asset_id': ['nl0000301109', 'nl0000301109', 'gb00b03mlx29', 'gb00b03mlx29', 'lu0197800237', 'nl0000289965', np.nan], 'share': [1.0, 0.4, 0.6, 0.15, 0.6, 0.25, 1.0]}, columns=['household_id', 'asset_id', 'share']).set_index(['household_id', 'asset_id'])
    log_return = DataFrame({'asset_id': ['gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'lu0197800237', 'lu0197800237'], 't': [233, 234, 235, 180, 181], 'log_return': [0.09604978, -0.06524096, 0.03532373, 0.03025441, 0.036997]}).set_index(['asset_id', 't'])
    expected = DataFrame({'household_id': [2, 2, 2, 3, 3, 3, 3, 3], 'asset_id': ['gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'lu0197800237', 'lu0197800237'], 't': [233, 234, 235, 233, 234, 235, 180, 181], 'share': [0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.6, 0.6], 'log_return': [0.09604978, -0.06524096, 0.03532373, 0.09604978, -0.06524096, 0.03532373, 0.03025441, 0.036997]}).set_index(['household_id', 'asset_id', 't']).reindex(columns=['share', 'log_return'])
    result = merge(household.reset_index(), log_return.reset_index(), on=['asset_id'], how='inner').set_index(['household_id', 'asset_id', 't'])
    tm.assert_frame_equal(result, expected)
    expected = DataFrame({'household_id': [2, 2, 2, 3, 3, 3, 3, 3, 3, 1, 2, 4], 'asset_id': ['gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'gb00b03mlx29', 'lu0197800237', 'lu0197800237', 'nl0000289965', 'nl0000301109', 'nl0000301109', None], 't': [233, 234, 235, 233, 234, 235, 180, 181, None, None, None, None], 'share': [0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.6, 0.6, 0.25, 1.0, 0.4, 1.0], 'log_return': [0.09604978, -0.06524096, 0.03532373, 0.09604978, -0.06524096, 0.03532373, 0.03025441, 0.036997, None, None, None, None]}).set_index(['household_id', 'asset_id', 't']).reindex(columns=['share', 'log_return'])
    result = merge(household.reset_index(), log_return.reset_index(), on=['asset_id'], how='outer').set_index(['household_id', 'asset_id', 't'])
    tm.assert_frame_equal(result, expected)