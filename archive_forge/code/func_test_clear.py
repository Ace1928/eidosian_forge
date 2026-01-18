import pytest
import rpy2.robjects as robjects
import array
def test_clear():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 234
    assert len(env) == 2
    env.clear()
    assert len(env) == 0