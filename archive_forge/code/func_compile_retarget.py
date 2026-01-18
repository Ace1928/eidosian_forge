import abc
import weakref
from numba.core import errors
@abc.abstractmethod
def compile_retarget(self, orig_disp):
    """Returns the retargeted dispatcher.

        See numba/tests/test_retargeting.py for example usage.
        """
    pass