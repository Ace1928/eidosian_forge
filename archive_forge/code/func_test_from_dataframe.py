import numpy as np
import pandas
import pytest
from modin_spreadsheet import SpreadsheetWidget
import modin.experimental.spreadsheet as mss
import modin.pandas as pd
def test_from_dataframe():
    data = get_test_data()
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = mss.from_dataframe(modin_df)
    assert isinstance(modin_result, SpreadsheetWidget)
    with pytest.raises(TypeError):
        mss.from_dataframe(pandas_df)

    def can_edit_row(row):
        return row['D'] > 2
    modin_result = mss.from_dataframe(modin_df, show_toolbar=True, show_history=True, precision=1, grid_options={'forceFitColumns': False, 'filterable': False}, column_options={'D': {'editable': True}}, column_definitions={'editable': False}, row_edit_callback=can_edit_row)
    assert isinstance(modin_result, SpreadsheetWidget)