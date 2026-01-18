import pytest
from pandas.util._validators import validate_args_and_kwargs
def test_duplicate_argument(_fname):
    min_fname_arg_count = 2
    compat_args = {'foo': None, 'bar': None, 'baz': None}
    kwargs = {'foo': None, 'bar': None}
    args = (None,)
    msg = f"{_fname}\\(\\) got multiple values for keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)