from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('kwargs,msg', [({'index_col': ['foo', 'bar']}, 'index_col must only contain row numbers when specifying a multi-index header'), ({'index_col': [0, 1], 'names': ['foo', 'bar']}, 'cannot specify names when specifying a multi-index header'), ({'index_col': [0, 1], 'usecols': ['foo', 'bar']}, 'cannot specify usecols when specifying a multi-index header')])
def test_header_multi_index_invalid(all_parsers, kwargs, msg):
    data = 'C0,,C_l0_g0,C_l0_g1,C_l0_g2\n\nC1,,C_l1_g0,C_l1_g1,C_l1_g2\nC2,,C_l2_g0,C_l2_g1,C_l2_g2\nC3,,C_l3_g0,C_l3_g1,C_l3_g2\nR0,R1,,,\nR_l0_g0,R_l1_g0,R0C0,R0C1,R0C2\nR_l0_g1,R_l1_g1,R1C0,R1C1,R1C2\nR_l0_g2,R_l1_g2,R2C0,R2C1,R2C2\nR_l0_g3,R_l1_g3,R3C0,R3C1,R3C2\nR_l0_g4,R_l1_g4,R4C0,R4C1,R4C2\n'
    parser = all_parsers
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=[0, 1, 2, 3], **kwargs)