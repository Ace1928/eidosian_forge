import operator
import sys
import warnings
def quick_grad_check(fun, arg0, extra_args=(), kwargs={}, verbose=True, eps=0.0001, rtol=0.0001, atol=1e-06, rs=None):
    warnings.warn(deprecation_msg)
    from autograd.test_util import check_grads
    fun_ = lambda arg0: fun(arg0, *extra_args, **kwargs)
    check_grads(fun_, modes=['rev'], order=1)(arg0)