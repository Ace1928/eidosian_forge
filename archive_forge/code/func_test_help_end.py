import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_help_end():
    tt.check_pairs(transform_and_reset(ipt.help_end), syntax['end_help'])