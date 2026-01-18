from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_converters_type_must_be_dict(all_parsers):
    parser = all_parsers
    data = 'index,A,B,C,D\nfoo,2,3,4,5\n'
    if parser.engine == 'pyarrow':
        msg = "The 'converters' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), converters=0)
        return
    with pytest.raises(TypeError, match='Type converters.+'):
        parser.read_csv(StringIO(data), converters=0)