from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
def test_center_ljust_rjust_fillchar_bad_arg_raises(any_string_dtype):
    s = Series(['a', 'bb', 'cccc', 'ddddd', 'eeeeee'], dtype=any_string_dtype)
    template = 'fillchar must be a character, not {dtype}'
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        s.str.center(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        s.str.ljust(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='str')):
        s.str.rjust(5, fillchar='XY')
    with pytest.raises(TypeError, match=template.format(dtype='int')):
        s.str.center(5, fillchar=1)
    with pytest.raises(TypeError, match=template.format(dtype='int')):
        s.str.ljust(5, fillchar=1)
    with pytest.raises(TypeError, match=template.format(dtype='int')):
        s.str.rjust(5, fillchar=1)