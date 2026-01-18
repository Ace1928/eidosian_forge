import sys
import greenlet
from greenlet.tests import _test_extension_cpp
def run_unhandled_exception_in_greenlet_aborts():

    def _():
        _test_extension_cpp.test_exception_switch_and_do_in_g2(_test_extension_cpp.test_exception_throw_nonstd)
    g1 = greenlet.greenlet(_)
    g1.switch()