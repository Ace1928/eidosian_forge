import pytest
from trio import run
from trio.lowlevel import RunVar, RunVarToken
from ... import _core
def test_accessing_runvar_outside_run_call_fails() -> None:
    t1 = RunVar[str]('test1')
    with pytest.raises(RuntimeError):
        t1.set('asdf')
    with pytest.raises(RuntimeError):
        t1.get()

    async def get_token() -> RunVarToken[str]:
        return t1.set('ok')
    token = run(get_token)
    with pytest.raises(RuntimeError):
        t1.reset(token)