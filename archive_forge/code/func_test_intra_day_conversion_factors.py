import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
@pytest.mark.parametrize('freq1,freq2,expected', [('D', 'H', 24), ('D', 'T', 1440), ('D', 'S', 86400), ('D', 'L', 86400000), ('D', 'U', 86400000000), ('D', 'N', 86400000000000), ('H', 'T', 60), ('H', 'S', 3600), ('H', 'L', 3600000), ('H', 'U', 3600000000), ('H', 'N', 3600000000000), ('T', 'S', 60), ('T', 'L', 60000), ('T', 'U', 60000000), ('T', 'N', 60000000000), ('S', 'L', 1000), ('S', 'U', 1000000), ('S', 'N', 1000000000), ('L', 'U', 1000), ('L', 'N', 1000000), ('U', 'N', 1000)])
def test_intra_day_conversion_factors(freq1, freq2, expected):
    assert period_asfreq(1, get_freq_code(freq1), get_freq_code(freq2), False) == expected