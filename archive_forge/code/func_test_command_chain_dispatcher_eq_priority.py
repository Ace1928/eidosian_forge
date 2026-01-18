import pytest
from IPython.core.error import TryNext
from IPython.core.hooks import CommandChainDispatcher
def test_command_chain_dispatcher_eq_priority():
    okay1 = Okay(u'okay1')
    okay2 = Okay(u'okay2')
    dp = CommandChainDispatcher([(1, okay1)])
    dp.add(okay2, 1)