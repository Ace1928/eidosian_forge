import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
@memory_no_filtered.setter
def memory_no_filtered(self, value):
    if bool(value):
        self.memory_no_filtered_mean = True
        self.memory_no_filtered_cov = True
    else:
        self.memory_no_filtered_mean = False
        self.memory_no_filtered_cov = False