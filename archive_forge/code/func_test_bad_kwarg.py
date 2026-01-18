import pytest
from pandas.util._validators import (
def test_bad_kwarg(_fname):
    good_arg = 'f'
    bad_arg = good_arg + 'o'
    compat_args = {good_arg: 'foo', bad_arg + 'o': 'bar'}
    kwargs = {good_arg: 'foo', bad_arg: 'bar'}
    msg = f"{_fname}\\(\\) got an unexpected keyword argument '{bad_arg}'"
    with pytest.raises(TypeError, match=msg):
        validate_kwargs(_fname, kwargs, compat_args)