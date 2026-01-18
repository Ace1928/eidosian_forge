import pytest
import rpy2.robjects as robjects
import array
def test_getsetitem():
    env = robjects.Environment()
    env['a'] = 123
    assert 'a' in env
    a = env['a']
    assert len(a) == 1
    assert a[0] == 123