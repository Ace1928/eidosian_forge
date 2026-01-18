import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
def bind_fold_arguments(self, fold_arguments):
    """Bind the fold_arguments function
        """
    from numba.core.utils import chain_exception
    e = ForceLiteralArg(self.requested_args, fold_arguments, loc=self.loc)
    return chain_exception(e, self)