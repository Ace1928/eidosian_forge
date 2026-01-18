import pytest
from packaging.version import Version
from collections import namedtuple
from ...base import traits, File, TraitedSpec, BaseInterfaceInputSpec
from ..base import (
@pytest.mark.skipif(no_dipy() or Version(dipy_version()) < Version('1.4'), reason='DIPY >=1.4 required')
def test_get_default_args():
    from dipy.utils.deprecator import deprecated_params

    def test(dummy=11, x=3):
        return (dummy, x)

    @deprecated_params('x', None, '0.3', '0.5', alternative='test2.y')
    def test2(dummy=11, x=3):
        return (dummy, x)

    @deprecated_params(['dummy', 'x'], None, '0.3', alternative='test2.y')
    def test3(dummy=11, x=3):
        return (dummy, x)

    @deprecated_params(['dummy', 'x'], None, '0.3', '0.5', alternative='test2.y')
    def test4(dummy=11, x=3):
        return (dummy, x)
    expected_res = {'dummy': 11, 'x': 3}
    for func in [test, test2, test3, test4]:
        assert get_default_args(func) == expected_res