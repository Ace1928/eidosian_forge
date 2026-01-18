import os
import subprocess
import contextlib
import functools
import tempfile
import shutil
import operator
import warnings
def passes(self, func):
    """
        Wrap func and replace the result with the truth
        value of the trap (True if no exception).

        First, give the decorator an alias to support Python 3.8
        Syntax.

        >>> passes = ExceptionTrap(ValueError).passes

        Now decorate a function that always fails.

        >>> @passes
        ... def fail():
        ...     raise ValueError('failed')

        >>> fail()
        False
        """
    return self.raises(func, _test=operator.not_)