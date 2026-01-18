import pytest
import inspect
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('value, expected', ((robjects.r('TRUE'), True), (robjects.r('1'), 1), (robjects.r('"abc"'), 'abc')))
def test_map_default_values(value, expected):
    assert robjects.functions._map_default_value(value) == expected