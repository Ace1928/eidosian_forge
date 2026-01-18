import sys
import warnings
import numpy as np
import param
from .. import util
from ..element import Element
from ..ndmapping import NdMapping
from .util import finite_range
@classmethod
def as_dframe(cls, dataset):
    """
        Returns the data of a Dataset as a dataframe avoiding copying
        if it already a dataframe type.
        """
    return dataset.dframe()