import pytest
from pandas.util._validators import validate_args_and_kwargs
def test_invalid_total_length_max_length_multiple(_fname):
    compat_args = ('foo', 'bar', 'baz')
    kwargs = {'foo': 'FOO', 'bar': 'BAR'}
    args = ('FoO', 'BaZ')
    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count
    msg = f'{_fname}\\(\\) takes at most {max_length} arguments \\({actual_length} given\\)'
    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)