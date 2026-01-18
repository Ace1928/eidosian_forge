from pytest import raises
from .. import deprecated
from ..deprecated import deprecated as deprecated_decorator
from ..deprecated import warn_deprecation
def test_deprecated_other_object(mocker):
    mocker.patch.object(deprecated, 'warn_deprecation')
    with raises(TypeError):
        deprecated_decorator({})