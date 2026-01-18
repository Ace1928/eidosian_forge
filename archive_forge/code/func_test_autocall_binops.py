import pytest
from IPython.core.prefilter import AutocallChecker
def test_autocall_binops():
    """See https://github.com/ipython/ipython/issues/81"""
    ip.magic('autocall 2')
    f = lambda x: x
    ip.user_ns['f'] = f
    try:
        assert ip.prefilter('f 1') == 'f(1)'
        for t in ['f +1', 'f -1']:
            assert ip.prefilter(t) == t
        pm = ip.prefilter_manager
        ac = AutocallChecker(shell=pm.shell, prefilter_manager=pm, config=pm.config)
        try:
            ac.priority = 1
            ac.exclude_regexp = '^[,&^\\|\\*/]|^is |^not |^in |^and |^or '
            pm.sort_checkers()
            assert ip.prefilter('f -1') == 'f(-1)'
            assert ip.prefilter('f +1') == 'f(+1)'
        finally:
            pm.unregister_checker(ac)
    finally:
        ip.magic('autocall 0')
        del ip.user_ns['f']