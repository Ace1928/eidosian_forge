import pytest
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
@pytest.mark.parametrize('func', ['cumsum', 'cumprod'])
def test_accumulators_disallowed(self, func):
    arr = DatetimeArray._from_sequence(['2000-01-01', '2000-01-02'], dtype='M8[ns]')._with_freq('infer')
    with pytest.raises(TypeError, match=f'Accumulation {func}'):
        arr._accumulate(func)