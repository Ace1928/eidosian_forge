import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('skipna', [True, False])
def test_accumulate_series(self, data, all_numeric_accumulations, skipna):
    op_name = all_numeric_accumulations
    ser = pd.Series(data)
    if self._supports_accumulation(ser, op_name):
        self.check_accumulate(ser, op_name, skipna)
    else:
        with pytest.raises((NotImplementedError, TypeError)):
            getattr(ser, op_name)(skipna=skipna)