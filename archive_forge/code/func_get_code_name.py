import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
def get_code_name(self, raw_code, transformed_code, number):
    """Compute filename given the code, and the cell number.

        Parameters
        ----------
        raw_code : str
            The raw cell code.
        transformed_code : str
            The executable Python source code to cache and compile.
        number : int
            A number which forms part of the code's name. Used for the execution
            counter.

        Returns
        -------
        The computed filename.
        """
    return code_name(transformed_code, number)