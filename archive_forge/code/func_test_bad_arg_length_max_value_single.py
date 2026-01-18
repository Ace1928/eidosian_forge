import pytest
from pandas.util._validators import validate_args
def test_bad_arg_length_max_value_single(_fname):
    args = (None, None)
    compat_args = ('foo',)
    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(args) + min_fname_arg_count
    msg = f'{_fname}\\(\\) takes at most {max_length} argument \\({actual_length} given\\)'
    with pytest.raises(TypeError, match=msg):
        validate_args(_fname, args, min_fname_arg_count, compat_args)