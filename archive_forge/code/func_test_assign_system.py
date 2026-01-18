import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def test_assign_system():
    tt.check_pairs(transform_and_reset(ipt.assign_from_system), syntax['assign_system'])