from pytest import raises
from .. import deprecated
from ..deprecated import deprecated as deprecated_decorator
from ..deprecated import warn_deprecation
def test_deprecated_decorator_text(mocker):
    mocker.patch.object(deprecated, 'warn_deprecation')

    @deprecated_decorator('Deprecation text')
    def my_func():
        return True
    result = my_func()
    assert result
    deprecated.warn_deprecation.assert_called_with('Call to deprecated function my_func (Deprecation text).')