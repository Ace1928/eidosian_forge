import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def autostrip(self, method):
    """
        Wrapper to strip each member of the output of `method`.

        Parameters
        ----------
        method : function
            Function that takes a single argument and returns a sequence of
            strings.

        Returns
        -------
        wrapped : function
            The result of wrapping `method`. `wrapped` takes a single input
            argument and returns a list of strings that are stripped of
            white-space.

        """
    return lambda input: [_.strip() for _ in method(input)]