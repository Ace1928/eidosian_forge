import functools
import operator
from matplotlib import _api
def checkSameFrame(self, rhs, func):
    """
        Check to see if frames are the same.

        = ERROR CONDITIONS
        - If the frame of the rhs Duration is not the same as our frame,
          an error is thrown.

        = INPUT VARIABLES
        - rhs     The Duration to check for the same frame
        - func    The name of the function doing the check.
        """
    if self._frame != rhs._frame:
        raise ValueError(f'Cannot {func} Durations with different frames.\nLHS: {self._frame}\nRHS: {rhs._frame}')