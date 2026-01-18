from pytest import raises
from .. import deprecated
from ..deprecated import deprecated as deprecated_decorator
from ..deprecated import warn_deprecation
def test_deprecated_class_text(mocker):
    mocker.patch.object(deprecated, 'warn_deprecation')

    @deprecated_decorator('Deprecation text')
    class X:
        pass
    result = X()
    assert result
    deprecated.warn_deprecation.assert_called_with('Call to deprecated class X (Deprecation text).')