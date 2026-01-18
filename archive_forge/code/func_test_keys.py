import pytest
import rpy2.robjects as robjects
import array
def test_keys():
    env = robjects.Environment()
    env['a'] = 123
    env['b'] = 234
    keys = list(env.keys())
    assert len(keys) == 2
    keys.sort()
    for it_a, it_b in zip(keys, ('a', 'b')):
        assert it_a == it_b