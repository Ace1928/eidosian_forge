import pytest
from pandas import Series
from pandas.core.strings.accessor import StringMethods
@pytest.fixture(params=_any_string_method, ids=ids)
def any_string_method(request):
    """
    Fixture for all public methods of `StringMethods`

    This fixture returns a tuple of the method name and sample arguments
    necessary to call the method.

    Returns
    -------
    method_name : str
        The name of the method in `StringMethods`
    args : tuple
        Sample values for the positional arguments
    kwargs : dict
        Sample values for the keyword arguments

    Examples
    --------
    >>> def test_something(any_string_method):
    ...     s = Series(['a', 'b', np.nan, 'd'])
    ...
    ...     method_name, args, kwargs = any_string_method
    ...     method = getattr(s.str, method_name)
    ...     # will not raise
    ...     method(*args, **kwargs)
    """
    return request.param