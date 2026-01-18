import pytest
import rpy2.rinterface as rinterface
import rpy2.rlike.container as rlc
@pytest.mark.parametrize('give_env', (True, False))
@pytest.mark.parametrize('use_rlock', (True, False))
def test_call_in_context(give_env, use_rlock):
    ls = rinterface.baseenv['ls']
    get = rinterface.baseenv['get']
    if give_env:
        env = rinterface.baseenv['new.env']()
    else:
        env = None
    assert 'foo' not in ls()
    with rinterface.local_context(env=env, use_rlock=use_rlock) as lc:
        lc['foo'] = 123
        assert tuple(get('foo')) == (123,)
    assert 'foo' not in ls()