from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
def test_join_on_single_col_dup_on_both(left_w_dups, right_w_dups):
    left_w_dups.join(right_w_dups, on='a', validate='many_to_many')
    msg = 'Merge keys are not unique in right dataset; not a many-to-one merge'
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(right_w_dups, on='a', validate='many_to_one')
    msg = 'Merge keys are not unique in left dataset; not a one-to-many merge'
    with pytest.raises(MergeError, match=msg):
        left_w_dups.join(right_w_dups, on='a', validate='one_to_many')