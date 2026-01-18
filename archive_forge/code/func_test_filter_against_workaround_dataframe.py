from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_filter_against_workaround_dataframe():
    letters = np.array(list(ascii_lowercase))
    N = 100
    random_letters = letters.take(np.random.default_rng(2).integers(0, 26, N, dtype=int))
    df = DataFrame({'ints': Series(np.random.default_rng(2).integers(0, 100, N)), 'floats': N / 10 * Series(np.random.default_rng(2).random(N)), 'letters': Series(random_letters)})
    grouped = df.groupby('ints')
    old_way = df[grouped.floats.transform(lambda x: x.mean() > N / 20).astype('bool')]
    new_way = grouped.filter(lambda x: x['floats'].mean() > N / 20)
    tm.assert_frame_equal(new_way, old_way)
    grouper = df.floats.apply(lambda x: np.round(x, -1))
    grouped = df.groupby(grouper)
    old_way = df[grouped.letters.transform(lambda x: len(x) < N / 10).astype('bool')]
    new_way = grouped.filter(lambda x: len(x.letters) < N / 10)
    tm.assert_frame_equal(new_way, old_way)
    grouped = df.groupby('letters')
    old_way = df[grouped.ints.transform(lambda x: x.mean() > N / 20).astype('bool')]
    new_way = grouped.filter(lambda x: x['ints'].mean() > N / 20)
    tm.assert_frame_equal(new_way, old_way)