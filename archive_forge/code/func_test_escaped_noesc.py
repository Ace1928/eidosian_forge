import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_escaped_noesc():
    tt.check_pairs(transform_and_reset(ipt.escaped_commands), syntax['escaped_noesc'])