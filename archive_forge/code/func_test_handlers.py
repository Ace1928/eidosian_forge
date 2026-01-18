from IPython.core import autocall
from IPython.testing import tools as tt
def test_handlers():
    call_idx = CallableIndexable()
    ip.user_ns['call_idx'] = call_idx
    run([('"no change"', '"no change"'), (u'lsmagic', "get_ipython().run_line_magic('lsmagic', '')")])
    autocallable = Autocallable()
    ip.user_ns['autocallable'] = autocallable
    ip.run_line_magic('autocall', '0')
    run([('len "abc"', 'len "abc"'), ('autocallable', 'autocallable()'), ('autocallable()', 'autocallable()')])
    ip.run_line_magic('autocall', '1')
    run([('len "abc"', 'len("abc")'), ('len "abc";', 'len("abc");'), ('len [1,2]', 'len([1,2])'), ('call_idx [1]', 'call_idx [1]'), ('call_idx 1', 'call_idx(1)'), ('len', 'len')])
    ip.run_line_magic('autocall', '2')
    run([('len "abc"', 'len("abc")'), ('len "abc";', 'len("abc");'), ('len [1,2]', 'len([1,2])'), ('call_idx [1]', 'call_idx [1]'), ('call_idx 1', 'call_idx(1)'), ('len', 'len()')])
    ip.run_line_magic('autocall', '1')
    assert failures == []