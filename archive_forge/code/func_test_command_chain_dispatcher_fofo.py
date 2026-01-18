import pytest
from IPython.core.error import TryNext
from IPython.core.hooks import CommandChainDispatcher
def test_command_chain_dispatcher_fofo():
    """Test a mixture of failing and succeeding hooks."""
    fail1 = Fail('fail1')
    fail2 = Fail('fail2')
    okay1 = Okay('okay1')
    okay2 = Okay('okay2')
    dp = CommandChainDispatcher([(0, fail1), (10, fail2), (15, okay2)])
    dp.add(okay1, 5)
    assert dp() == 'okay1'
    assert fail1.called is True
    assert okay1.called is True
    assert fail2.called is False
    assert okay2.called is False