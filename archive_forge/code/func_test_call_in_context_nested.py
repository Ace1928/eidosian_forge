import pytest
import rpy2.robjects as robjects
import array
@pytest.mark.parametrize('use_rlock', (True, False))
def test_call_in_context_nested(use_rlock):
    ls = robjects.baseenv['ls']
    get = robjects.baseenv['get']
    assert 'foo' not in ls()
    with robjects.environments.local_context() as lc_a:
        lc_a['foo'] = 123
        assert tuple(get('foo')) == (123,)
        with robjects.environments.local_context(use_rlock=use_rlock) as lc_b:
            lc_b['foo'] = 456
            assert tuple(get('foo')) == (456,)
        assert tuple(get('foo')) == (123,)
    assert 'foo' not in ls()