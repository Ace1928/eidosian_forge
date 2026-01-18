import numpy as np
import pandas as pd
import pandas._testing as tm
def test_apply_mutate_columns_multiindex():
    df = pd.DataFrame({('C', 'julian'): [1, 2, 3], ('B', 'geoffrey'): [1, 2, 3], ('A', 'julian'): [1, 2, 3], ('B', 'julian'): [1, 2, 3], ('A', 'geoffrey'): [1, 2, 3], ('C', 'geoffrey'): [1, 2, 3]}, columns=pd.MultiIndex.from_tuples([('A', 'julian'), ('A', 'geoffrey'), ('B', 'julian'), ('B', 'geoffrey'), ('C', 'julian'), ('C', 'geoffrey')]))

    def add_column(grouped):
        name = grouped.columns[0][1]
        grouped['sum', name] = grouped.sum(axis=1)
        return grouped
    msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        gb = df.groupby(level=1, axis=1)
    result = gb.apply(add_column)
    expected = pd.DataFrame([[1, 1, 1, 3, 1, 1, 1, 3], [2, 2, 2, 6, 2, 2, 2, 6], [3, 3, 3, 9, 3, 3, 3, 9]], columns=pd.MultiIndex.from_tuples([('geoffrey', 'A', 'geoffrey'), ('geoffrey', 'B', 'geoffrey'), ('geoffrey', 'C', 'geoffrey'), ('geoffrey', 'sum', 'geoffrey'), ('julian', 'A', 'julian'), ('julian', 'B', 'julian'), ('julian', 'C', 'julian'), ('julian', 'sum', 'julian')]))
    tm.assert_frame_equal(result, expected)